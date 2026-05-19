# vk_layer_tegra_x11_present

A Vulkan implicit layer that fixes broken Vulkan→X11 presentation on
NVIDIA Tegra L4T r32.x by routing presentation through GL/GLX instead of
the native Vulkan WSI path.

Built originally for Nintendo Switch running Lakka and L4T r32.x; should
work unchanged on any Tegra device on the r32.x BSP whose Vulkan driver
has the same WSI tearing problem.

## What it does

The NVIDIA Vulkan ICD shipped with L4T r32.x has a broken X11 WSI
implementation: presented frames tear regardless of present mode. The BSP
is end-of-life and there is no driver fix. The same driver, however,
provides:

- Vulkan external memory and semaphore export via
  `VK_KHR_external_memory_fd` and `VK_KHR_external_semaphore_fd`
  (OPAQUE_FD handle type)
- GL import of both via `GL_EXT_memory_object_fd` and `GL_EXT_semaphore_fd`
- Working vsync-locked GL presentation via `GLX_SGI_video_sync`

This layer plumbs the three together. The application's Vulkan rendering
is untouched. We replace the WSI surface and swapchain with our own
implementation that:

1. Allocates swapchain images as OPAQUE_FD-exportable Vulkan images.
2. Imports each image into a GL/GLX context as a GL texture.
3. At `vkQueuePresentKHR` time: bridges the application's render-done
   semaphore into a GL semaphore and hands the image to a per-swapchain
   worker thread.
4. The worker samples the image into the GLX backbuffer and calls
   `glXSwapBuffers(interval=1)`, which provides vsync-locked presentation
   in all environments. If the swap returned quickly (direct scanout —
   no compositor, or compositor bypassed), the worker also calls
   `glXWaitVideoSyncSGI` to sleep the thread in the kernel until the next
   vblank rather than spinning. If the swap was slow (a compositor blocked
   it for a full vblank), SGI is skipped — the vblank was already consumed.
5. Bridges GL's sample-done back into a Vulkan semaphore so that the
   next `vkAcquireNextImageKHR` correctly gates the application's re-use
   of the image.

No EGL, no dmabuf, no DRM, no custom kernel modules. Only the Vulkan↔GL
interop primitives that the stock driver provides.

## Status

Working on Lakka and Switchroot Ubuntu Noble L4T r32.x Switch builds.

Tested:

- vkcube — clean 60Hz vsync
- RetroArch with various Vulkan-capable cores (Dolphin etc.) — clean,
  no tearing, expected idle CPU
- Sascha Willems Vulkan samples (triangle, bloom, and the rest of the
  suite) — work normally; the symbol-export tightening in v2 fixed an
  earlier loader-deadlock with these
- PPSSPP libretro core — works for simple titles; some titles fault, see
  KNOWN ISSUES

## Architecture

```
Application's render thread                       Worker thread (per swapchain)
─────────────────────────                         ──────────────────────────
vkAcquireNextImageKHR
  ├── bridge submit: wait gl_sample[i], signal acquire_sem,
  │   signal acquire_fence
  └── WaitForFences(acquire_fence)   ← backpressure                    ←┐
                                                                        │
[app renders into image i]                                              │
                                                                        │
vkQueuePresentKHR                                                       │
  ├── bridge submit: wait app_render_done, signal vk_render_done[i]     │
  └── post {idx=i} to worker mailbox                  → take work       │
                                                       glWaitSemaphoreEXT(vk_render_done)
                                                       draw textured quad
                                                       glSignalSemaphoreEXT(gl_sample[i])  ─┘
                                                       glXSwapBuffers(interval=1)  ← vsync gate
                                                       [if swap was fast]:
                                                         glXWaitVideoSyncSGI  ← kernel sleep
                                                       mark mailbox free
```

The worker thread owns the GLX context for the swapchain's entire
lifetime and opens its own Xlib `Display*` (Xlib is not thread-safe to
share even with `XInitThreads`).

The backpressure point is the `WaitForFences` in
`vkAcquireNextImageKHR`: the fence signals only after the worker has
actually consumed the image through GL. The app's render loop is paced
to real presentation rate through this fence.

## Vsync and CPU usage

**Vsync.** The worker uses `glXSwapBuffers(interval=1)` for all FIFO
present modes. This is exactly what any standard GL application does,
and it provides correct vsync-locked presentation in every environment:

- *Without a compositor* (Lakka, direct-to-display): the driver schedules
  an async DRM page flip and returns immediately. The display controller
  presents the buffer at the next vblank.
- *With a compositor* (KWin, Mutter, picom etc.): the compositor pipeline
  blocks `glXSwapBuffers` for approximately one vblank while it takes
  ownership of the buffer, then presents it through its own rendering
  loop. Vsync is implicit in the compositor's cadence.

In both cases one `glXSwapBuffers(interval=1)` call = one frame at the
display's native refresh rate, with no tearing.

**CPU usage.** On NVIDIA Tegra, when there is no compositor the driver's
`glXSwapBuffers` returns immediately (async flip) rather than waiting
for the vblank. The worker thread would then spin at the top of its loop
until the app delivers the next frame, burning CPU.

To keep the thread sleeping in the kernel between frames, the layer
conditionally calls `glXWaitVideoSyncSGI` after each swap. The decision
is made by measuring how long `glXSwapBuffers` itself took (EWMA over
~8 frames):

- **Swap was fast** (< 1/4 vblank period): direct scanout — the vblank
  hasn't been consumed yet. Call `glXWaitVideoSyncSGI` to sleep the
  thread until the next vblank.
- **Swap was slow** (≥ 1/4 vblank period): a compositor blocked the swap
  for a full vblank. The vblank is already consumed; calling SGI would
  add a *second* vblank wait and cap throughput at 1/2 the native display framerate. Skip SGI.

This requires no knowledge of whether a compositor is running, no X
property queries, and no static configuration. It adapts automatically
within ~8 frames whenever the compositing state changes (compositor
starts, stops, or a window gains/loses compositor bypass).

If `GLX_SGI_video_sync` is unavailable at runtime (uncommon on NVIDIA
proprietary), the layer falls back to swap-interval-1 alone — vsync is
still correct, but the worker may show elevated CPU when running without
a compositor. The log records which path is active whenever it changes.

## Swap pipeline ordering

Before each `glXSwapBuffers` the worker calls `glFinish()`, and after
the swap it calls `XFlush()` on the worker's Display. Both matter:

`glFinish` blocks until all queued GL commands actually complete on the
GPU. With `glFlush` alone (which only kicks the command queue), the
swap-at-vblank request in `glXSwapBuffers` can be queued while the
sample draw is still in flight; the driver's vblank logic then races
with the rendering. On the internal Switch panel this is harmless; on
the docked HDMI output it manifests as a slowly-drifting tear line
despite swap interval 1 being honored.

`XFlush` pushes the swap request out of Xlib's output buffer onto the
wire to the X server immediately, rather than letting it sit until the
next X call (which in steady-state is `XGetGeometry` ~16ms later).
Without it, the X server doesn't see the swap request until after the
vblank it should have been scheduled for has already passed, also
causing tearing.

The cost of `glFinish` is one CPU stall per frame — sub-millisecond on
a single textured quad. The cost of `XFlush` is one syscall. Both
together close the worker's swap pipeline ordering tightly enough that
the docked output sees the same clean vsync behavior as the internal
panel.

## Runtime library loading

The `.so` does NOT link libGL, libGLX, or libX11. It depends only on
libc, libdl, and libpthread.

This is deliberate. The Vulkan loader dlopens implicit layers while
holding an internal mutex during `vkCreateInstance`. If our DT_NEEDED
list included libGL, ld.so would run libGL's constructor as part of the
load, with the loader's mutex held, and on some loader/driver
combinations that constructor calls back into the Vulkan loader (e.g.,
glvnd vendor registration) — which then tries to acquire the
already-held mutex, and the whole vkCreateInstance hangs.

The fix: `dlopen()` libGL / libGLX / libX11 ourselves at swapchain
creation time, well past the loader-mutex window, and resolve every
function via `dlsym` into a small table. The call sites use `#define`
macros to redirect transparently — `glXSwapBuffers(...)` expands to
`g_libs.glXSwapBuffers(...)`.

## Symbol export

Only `vkNegotiateLoaderLayerInterfaceVersion` is exported from the
`.so`. `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr` are NOT
exported.

Why: under Vulkan layer interface v2, the loader stores the layer's
GIPA and GDPA function pointers from the `VkNegotiateLayerInterface`
struct filled in by negotiate, and calls them directly. The loader does
not need those symbols exported by name.

But if we DO export them, the loader's ICD-load path can do
`dlsym(RTLD_DEFAULT, "vkGetInstanceProcAddr")` and find OUR symbol
instead of the loader's own. The loader then re-enters our GIPA looking
for `vkCreateInstance`, gets our `layer_CreateInstance`, calls it —
while the outer `vkCreateInstance` is still on the stack with the
loader mutex held. Recursive deadlock.

The fix is one line in the version script: only export
`vkNegotiateLoaderLayerInterfaceVersion`. Confirmed with
`nm -D --defined-only libVkLayer_tegra_x11_present.so`.

## Vulkan queue serialization

The application's `vkQueueSubmit` calls and the layer's internal bridge
submits are serialized through a per-device mutex. This is required by
the Vulkan spec (external synchronization on the queue parameter) and
matters when the application uses a separate render thread (PPSSPP,
Citra, etc.). The cost is negligible — a mutex acquire per submit. Apps
that submit from a single thread pay nothing.

## Compositor coexistence

The layer works correctly whether a compositor is running or not, and
adapts automatically between the two cases.

**With a compositor (KWin, Mutter, picom etc.):** `glXSwapBuffers`
blocks for approximately one vblank while the compositor takes the
buffer. The layer detects this from the measured swap duration (EWMA
over ~8 frames) and skips the `glXWaitVideoSyncSGI` call that would
otherwise add a second vblank wait and halve throughput. The compositor
presents the frame through its own vsync-locked pipeline.

Alpha channel, CSD shadows, translucent borders, and compositor blur
effects all work: the layer picks a 32-bit ARGB GLX visual for the
child window so the compositor can blend it against the desktop.

**Without a compositor (Lakka, direct-to-display):** `glXSwapBuffers`
returns immediately after scheduling an async DRM flip. The layer
detects the fast return and calls `glXWaitVideoSyncSGI` to sleep the
worker thread in the kernel until the next vblank, keeping CPU usage
low. Vsync timing is end-to-end accurate with no intermediate pipeline.

**Transitions:** if a compositor starts, stops, or a window gains or
loses compositor bypass (`_NET_WM_BYPASS_COMPOSITOR`) after the
swapchain is created, the layer detects the change within ~8 frames
and switches paths automatically. A log line at info level records each
transition and the average swap duration that triggered it.

## Layout transition handling

The Vulkan WSI defines `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` for swapchain
images. Real WSI implementations track that as a hardware-meaningful
state. We don't have a real swapchain — our images are normal Vulkan
images with no presentation layout — so any pipeline barrier or render
pass that transitions our images to/from `PRESENT_SRC_KHR` would fault
the GPU on this driver.

The layer intercepts `vkCmdPipelineBarrier`, `vkCreateRenderPass`, and
`vkCreateRenderPass2KHR` and rewrites any `PRESENT_SRC_KHR` layout on a
managed image to `SHADER_READ_ONLY_OPTIMAL`. Application code is
unchanged; the rewrite is invisible.

## Environment variables

| Variable | Effect |
|---|---|
| `VK_TEGRA_X11_PRESENT_DISABLE=1` | Passthrough mode — the layer loads but forwards every call unmodified. Useful for A/B testing against the native (broken) WSI. Honored by the Vulkan loader via `disable_environment` in the manifest. |
| `VK_TEGRA_X11_PRESENT_LOG=N` | Log verbosity: 0=silent, 1=warn/err (default), 2=info, 3=debug. |
| `VK_TEGRA_X11_PRESENT_LOG_FILE=PATH` | Append logs to `PATH` in addition to stderr. |
| `VK_TEGRA_X11_PRESENT_DIAG=1` | Diagnostic mode. After every `vkQueueSubmit`, the layer calls `vkDeviceWaitIdle` and reports if the GPU faulted on that submit. Catastrophically slow; one-shot use only for fault localization. |

## Build and install

```sh
make
sudo make install
```

This installs the `.so` to `/usr/lib/aarch64-linux-gnu/` and the layer
JSON to `/usr/share/vulkan/implicit_layer.d/`. The Vulkan loader picks
the layer up automatically — no per-app environment variable required.

For non-multiarch layouts (Lakka, similar embedded distros), override
the library install path:

```sh
make install LAYERLIBDIR=/usr/lib
```

The Lakka build recipe does this automatically.

### Build dependencies

- A C compiler (gcc or clang)
- Vulkan headers (`vulkan/vulkan.h`, `vulkan/vk_layer.h`, `vulkan/vk_icd.h`)
- GL/GLX/X11 headers (for the type declarations only — these libraries
  are not linked at build time)

### Runtime dependencies

- libc, libdl, libpthread (DT_NEEDED)
- libGL / libGLX / libX11 (dlopen'd at swapchain creation; must be
  present at runtime)
- Vulkan loader 1.1+ with layer interface v2 support
- An NVIDIA Tegra Vulkan ICD (or any ICD that supports the OPAQUE_FD
  external memory/semaphore extensions and `GLX_SGI_video_sync`)

## Known issues

None confirmed against v5 at time of release.

The earlier PPSSPP libretro core launch crash was resolved in v2 by
adding the per-device queue submission mutex for spec-compliant
external synchronization. Games launch and run.

If a concrete reproduction of any new bug shows up, the first step is
the control comparison — does the issue reproduce with the layer
disabled?

```sh
# With layer (normal):
DISPLAY=:0 VK_TEGRA_X11_PRESENT_LOG=2 \
    VK_TEGRA_X11_PRESENT_LOG_FILE=/tmp/with-layer.log <command>

# Without layer (the loader skips our manifest):
DISPLAY=:0 VK_TEGRA_X11_PRESENT_DISABLE=1 <command>
```

If both runs reproduce the issue, it isn't the layer — report upstream
to the app or the ICD vendor. If only the with-layer run fails, it's
a real layer bug.

## Done in v5

- **Compositor-aware vsync strategy.** Previously the layer always
  used `glXSwapBuffers(interval=1)` followed by `glXWaitVideoSyncSGI`.
  Under a running X compositor (KWin, Mutter, picom etc.) the
  interval-1 swap blocks for a full vblank while the compositor takes
  ownership of the buffer, and the SGI wait then consumes a second
  vblank — net throughput 30 Hz instead of 60 Hz.

  v5 fixes this with an adaptive approach that requires no static
  compositor detection. `glXSwapBuffers(interval=1)` is always correct
  — it provides vsync in both compositor and non-compositor
  environments, just as any standard GL application uses it. The SGI
  call is a CPU-efficiency add-on that is only valid when the swap
  returned quickly (no compositor consumed the vblank). The layer
  measures `glXSwapBuffers` duration per-frame (EWMA over ~8 frames)
  and skips the SGI call whenever the average duration indicates the
  compositor is blocking the swap. This handles every compositor
  configuration — including `_NET_WM_BYPASS_COMPOSITOR` being set or
  cleared, the compositor starting or stopping mid-session, and KDE's
  "Allow applications to block compositing" setting — without any X
  property queries or prior configuration.

  The previous `set_bypass_compositor` call has been removed: the
  layer now cooperates with the compositor so window shadows, blur, and
  CSD decorations work correctly on desktop deployments.

## Done in v4

- **`VK_PRESENT_MODE_MAILBOX_KHR` now has the right semantics.** The
  worker's mailbox accepts new presents non-blockingly: when a present
  arrives while the slot still has a previous unconsumed image, the
  new image replaces the old one and the caller is told which image
  was displaced. The caller then issues a one-shot cleanup bridge
  submit that consumes the displaced image's `vk_render_done`
  semaphore (which had already been signalled for the dropped frame)
  and signals its `gl_sample_done` semaphore so the next Acquire of
  that image succeeds.

  FIFO mode is unchanged — still blocks the producer when the slot
  is full, which is the right behaviour for vsync-locked apps.

  The worker still paces presentation at vblank rate in MAILBOX
  mode. MAILBOX changes only the *producer-side blocking* behaviour;
  the consumer still presents one image per display refresh.

## Done in v3

- **`VK_GOOGLE_display_timing`** is implemented. Apps that use the
  extension to compute frame-drop metrics (Chromium, GeForce NOW
  client, some game engines) receive accurate per-present timing data
  instead of falling back to the over-reporting heuristic.

  Implementation:
  - The layer advertises `VK_GOOGLE_display_timing` v1 via the manifest's
    `device_extensions` list.
  - `CreateDevice` strips this layer-provided extension from the
    extension list before forwarding to the ICD (the ICD doesn't
    implement it).
  - `vkGetRefreshCycleDurationGOOGLE` returns the swapchain's measured
    refresh duration. The value defaults to 1/60 Hz and is refined
    online by an EWMA over inter-vblank intervals filtered to a
    [50 Hz, 75 Hz] sanity range.
  - `vkGetPastPresentationTimingGOOGLE` returns up to 64 most-recent
    entries from a ring written by the worker thread immediately after
    each `glXWaitVideoSyncSGI` returns.
  - `VkPresentTimesInfoGOOGLE` is parsed out of `vkQueuePresentKHR`'s
    pNext chain; `presentID` and `desiredPresentTime` are plumbed to
    the worker through the mailbox and recorded in the ring.
  - Presents without a presentID (apps not using the extension) are
    not recorded, keeping the ring focused on what the app actually
    wants to track.

## Alpha and compositor blending

For apps with client-side decorations and translucent shadows (Chromium,
Electron, GTK3/4 CSD apps, Qt apps with native blur, etc.), the layer
needs to present an RGBA window that the X compositor can alpha-blend.
`pick_fbconfig` picks a 32-bit ARGB visual when one is available, and
the fragment shader writes the full RGBA from the Vulkan-rendered image
so the alpha channel is preserved through the GL→GLX→X-server pipeline.

If a 32-bit visual isn't available (rare; almost all NVIDIA GLX
configurations expose one), the layer falls back to a 24-bit RGB visual
— the window will be opaque and any CSD shadow region will show as
opaque black. The actual rendering is unaffected.

## Files

```
vk_layer_tegra_x11_present.c     main source
vk_layer_tegra_x11_present.json  Vulkan loader manifest
vk_layer_tegra_x11_present.map   linker version script (export control)
Makefile                         build and install
COPYING                          full GPL-2.0 license text
```

## Credits

Authors:

- **gavin_darkglider** — project owner, original concept, Lakka and
  upstream Tegra integration, deployment.
- **theofficialgman** — compositor-aware vsync strategy and direct/composited
  swap-path separation; broader Tegra Vulkan/X11 expertise.
- **Anthropic Claude (AI assistant)** — implementation work across the
  layer's architecture and iteration cycles.

Honorable mentions, without whose work the platform this layer runs on
wouldn't exist:

- **Ctcaer** — roughly 90% of the kernel work for Switch L4T.
- **Azkali**
- **makinbacon**

Thanks to testers who put real hardware time into shaking out bugs that
no synthetic test would have caught:

- **Angel_hp**
- **Zenjiki**
- **Kinro**

## License

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version. See `COPYING` for the full
license text.

Source files carry an `SPDX-License-Identifier: GPL-2.0-or-later` tag
in their header for machine-readable license tooling.
