/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Copyright (C) 2026 gavin_darkglider
 * Copyright (C) 2026 theofficialgman
 * Copyright (C) 2026 Anthropic (Claude AI assistant contributions)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

/*
 * vk_layer_tegra_x11_present.c
 *
 * Vulkan implicit layer for NVIDIA Tegra L4T r32.x that fixes the broken
 * Vulkan-on-X11 present path by routing presentation through GL/GLX
 * instead of Vulkan WSI.
 *
 * BACKGROUND
 *
 * On Tegra L4T r32.x the Nvidia Vulkan ICD's WSI implementation for X11
 * does not produce vsync-locked presentation. Frames tear. This is a known
 * driver-side issue with no available fix from Nvidia (the BSP is EOL'd).
 *
 * However, the same driver does support:
 *   - Vulkan external memory export via VK_KHR_external_memory_fd
 *     (OPAQUE_FD handle type, OPTIMAL tiling, BGRA8/RGBA8 UNORM/SRGB)
 *   - Vulkan external semaphore export via VK_KHR_external_semaphore_fd
 *     (OPAQUE_FD handle type)
 *   - GL import of both via GL_EXT_memory_object_fd and GL_EXT_semaphore_fd
 *   - Working vsync-locked GL presentation through GLX_SGI_video_sync.
 *
 * This layer plumbs the two together. The application's Vulkan rendering is
 * untouched; we replace the WSI surface and swapchain with our own
 * implementation that:
 *
 *   1. Allocates the swapchain images as OPAQUE_FD-exportable Vulkan images
 *      (the application renders into them as if they were normal swapchain
 *      images).
 *   2. Imports each image into our GL/GLX context as a GL texture.
 *   3. At vkQueuePresentKHR time: bridges the application's render-done
 *      semaphore into a GL semaphore and posts the image to our worker.
 *   4. Bridges GL's sample-done back into a Vulkan semaphore so that the
 *      next vkAcquireNextImageKHR correctly gates the application's
 *      re-use of the image.
 *
 * No EGL, no dmabuf, no DRM. Only Nvidia's own Vulkan↔GL interop primitives.
 *
 * ARCHITECTURE
 *
 * One worker thread per swapchain owns the GLX context for the lifetime of
 * the swapchain. The application's render thread calls Acquire/Present;
 * Present hands work to the worker via a single-slot mailbox and returns
 * immediately. The worker samples the image into the GLX backbuffer, calls
 * glXSwapBuffers (with swap interval 0), then blocks on the actual hardware
 * vblank via glXWaitVideoSyncSGI. This pattern lets the worker thread sleep
 * in the kernel during vsync, rather than spinning in libGLX_nvidia's
 * sched_yield-based default wait. The technique is the same one KWin uses
 * for NVIDIA on X11 (see plugins/platforms/x11/standalone/glxbackend.cpp,
 * SGIVideoSyncVsyncMonitor).
 *
 * vkAcquireNextImageKHR CPU-blocks the application on a per-image fence
 * that the worker signals via the GL→Vulkan semaphore bridge; this is what
 * paces the app's render loop to real presentation rate.
 *
 * RUNTIME LIBRARY LOADING
 *
 * The .so does not link libGL, libGLX, or libX11. The Vulkan loader holds
 * an internal mutex during vkCreateInstance and dlopen()s implicit layers;
 * if our DT_NEEDED listed libGL, that library's constructor would run with
 * the loader mutex held, and on some systems that constructor calls back
 * into the loader, recursively deadlocking. We dlopen GL/X11 ourselves at
 * CreateSwapchain time (well past the loader-mutex window) and resolve
 * every function via dlsym into a table. See the lib_load() block.
 *
 * SYMBOL EXPORT
 *
 * Only vkNegotiateLoaderLayerInterfaceVersion is exported from the .so.
 * vkGetInstanceProcAddr and vkGetDeviceProcAddr are NOT — exporting them
 * causes the Vulkan loader's ICD-load path to resolve our symbols via
 * dlsym(RTLD_DEFAULT, "vkGetInstanceProcAddr") and re-enter our layer
 * recursively while the loader mutex is still held, deadlocking
 * vkCreateInstance. The loader doesn't need these symbols exported under
 * interface v2; it gets the function pointers from the
 * VkNegotiateLayerInterface struct that our negotiate fills in.
 *
 * BYPASS
 *
 * Setting the environment variable VK_TEGRA_X11_PRESENT_DISABLE=1 turns
 * the layer into a transparent passthrough. Useful for A/B testing the
 * native WSI against the layer-redirected one.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>

#define VK_USE_PLATFORM_XLIB_KHR 1
#define VK_USE_PLATFORM_XCB_KHR  1
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vk_icd.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <X11/Xlib.h>
#include <X11/Xlib-xcb.h>
#include <X11/Xatom.h>
#include <xcb/xcb.h>
#include <dlfcn.h>

/* RUNTIME LIBRARY LOADING
 *
 * We deliberately do NOT link against libGL, libGLX, or libX11 at build
 * time. The reason: implicit Vulkan layers are dlopen()'d by the Vulkan
 * loader while it holds an internal mutex during vkCreateInstance. If our
 * .so has DT_NEEDED entries for libGL, ld.so will recursively load libGL
 * before our constructors run, and libGL's own constructor (via the glvnd
 * dispatch path) calls back into the Vulkan loader to register vendor
 * support. That callback tries to acquire the same loader mutex that the
 * outer vkCreateInstance already holds — recursive deadlock.
 *
 * Observed symptom: Sascha Willems Vulkan samples hang inside
 * vkCreateInstance with a backtrace showing recursive entry into the
 * loader's createInstance through libGL's init path. RetroArch and vkcube
 * don't hit it because they happen to load libGL via other paths before
 * vkCreateInstance fires.
 *
 * Workaround: dlopen libGL/libGLX/libX11 ourselves at CreateSwapchain time
 * (when no loader mutex is held) and call every function through a table
 * of dlsym'd function pointers. The .so itself depends only on libc,
 * libdl, libpthread, and libvulkan headers (Vulkan calls go through the
 * layer dispatch, never linked).
 *
 * The macros after the LibTable struct redirect all GL/GLX/X11 calls in
 * the rest of this file to go through the table — minimal source-level
 * impact, but the link-time dependency vanishes. */

#define X11_FUNCS(M) \
    M(XInitThreads,      Status,  (void)) \
    M(XOpenDisplay,      Display*,(const char *)) \
    M(XCloseDisplay,     int,     (Display *)) \
    M(XInternAtom,       Atom,    (Display *, const char *, Bool)) \
    M(XCreateColormap,   Colormap,(Display *, Window, Visual *, int)) \
    M(XCreateWindow,     Window,  (Display *, Window, int, int, unsigned, unsigned, unsigned, int, unsigned, Visual *, unsigned long, XSetWindowAttributes *)) \
    M(XDestroyWindow,    int,     (Display *, Window)) \
    M(XFreeColormap,     int,     (Display *, Colormap)) \
    M(XFree,             int,     (void *)) \
    M(XMapWindow,        int,     (Display *, Window)) \
    M(XGetGeometry,      Status,  (Display *, Drawable, Window *, int *, int *, unsigned *, unsigned *, unsigned *, unsigned *)) \
    M(XGetWindowAttributes, Status, (Display *, Window, XWindowAttributes *)) \
    M(XResizeWindow,     int,     (Display *, Window, unsigned, unsigned)) \
    M(XFlush,            int,     (Display *)) \
    M(XChangeProperty,   int,     (Display *, Window, Atom, Atom, int, int, const unsigned char *, int))

#define GL_FUNCS(M) \
    M(glGetError,                GLenum,  (void)) \
    M(glClear,                   void,    (GLbitfield)) \
    M(glClearColor,              void,    (GLfloat, GLfloat, GLfloat, GLfloat)) \
    M(glViewport,                void,    (GLint, GLint, GLsizei, GLsizei)) \
    M(glActiveTexture,           void,    (GLenum)) \
    M(glBindTexture,             void,    (GLenum, GLuint)) \
    M(glGenTextures,             void,    (GLsizei, GLuint *)) \
    M(glDeleteTextures,          void,    (GLsizei, const GLuint *)) \
    M(glTexParameteri,           void,    (GLenum, GLenum, GLint)) \
    M(glDrawArrays,              void,    (GLenum, GLint, GLsizei)) \
    M(glBindVertexArray,         void,    (GLuint)) \
    M(glGenVertexArrays,         void,    (GLsizei, GLuint *)) \
    M(glDeleteVertexArrays,      void,    (GLsizei, const GLuint *)) \
    M(glBindBuffer,              void,    (GLenum, GLuint)) \
    M(glGenBuffers,              void,    (GLsizei, GLuint *)) \
    M(glDeleteBuffers,           void,    (GLsizei, const GLuint *)) \
    M(glBufferData,              void,    (GLenum, GLsizeiptr, const void *, GLenum)) \
    M(glEnableVertexAttribArray, void,    (GLuint)) \
    M(glVertexAttribPointer,     void,    (GLuint, GLint, GLenum, GLboolean, GLsizei, const void *)) \
    M(glCreateShader,            GLuint,  (GLenum)) \
    M(glDeleteShader,            void,    (GLuint)) \
    M(glShaderSource,            void,    (GLuint, GLsizei, const GLchar *const *, const GLint *)) \
    M(glCompileShader,           void,    (GLuint)) \
    M(glGetShaderiv,             void,    (GLuint, GLenum, GLint *)) \
    M(glGetShaderInfoLog,        void,    (GLuint, GLsizei, GLsizei *, GLchar *)) \
    M(glCreateProgram,           GLuint,  (void)) \
    M(glDeleteProgram,           void,    (GLuint)) \
    M(glAttachShader,            void,    (GLuint, GLuint)) \
    M(glLinkProgram,             void,    (GLuint)) \
    M(glGetProgramiv,            void,    (GLuint, GLenum, GLint *)) \
    M(glGetProgramInfoLog,       void,    (GLuint, GLsizei, GLsizei *, GLchar *)) \
    M(glUseProgram,              void,    (GLuint)) \
    M(glGetUniformLocation,      GLint,   (GLuint, const GLchar *)) \
    M(glUniform1i,               void,    (GLint, GLint)) \
    M(glFlush,                   void,    (void)) \
    M(glFinish,                  void,    (void))

#define GLX_FUNCS(M) \
    M(glXChooseFBConfig,         GLXFBConfig*, (Display *, int, const int *, int *)) \
    M(glXGetVisualFromFBConfig,  XVisualInfo*, (Display *, GLXFBConfig)) \
    M(glXCreateNewContext,       GLXContext,   (Display *, GLXFBConfig, int, GLXContext, Bool)) \
    M(glXDestroyContext,         void,         (Display *, GLXContext)) \
    M(glXMakeCurrent,            Bool,         (Display *, GLXDrawable, GLXContext)) \
    M(glXSwapBuffers,            void,         (Display *, GLXDrawable)) \
    M(glXGetProcAddressARB,      __GLXextFuncPtr, (const GLubyte *))

/* Generate function-pointer typedefs for each entry. */
#define DECL_TYPEDEF(name, ret, args) typedef ret (*PFN_##name)args;
X11_FUNCS(DECL_TYPEDEF)
GL_FUNCS (DECL_TYPEDEF)
GLX_FUNCS(DECL_TYPEDEF)
#undef DECL_TYPEDEF

/* The pointer table, populated by lib_load(). */
typedef struct {
    void *handle_x11;
    void *handle_gl;
    void *handle_glx;
    bool  loaded;
#define DECL_FIELD(name, ret, args) PFN_##name name;
    X11_FUNCS(DECL_FIELD)
    GL_FUNCS (DECL_FIELD)
    GLX_FUNCS(DECL_FIELD)
#undef DECL_FIELD
} LibTable;

static LibTable g_libs;
static pthread_mutex_t g_libs_lock = PTHREAD_MUTEX_INITIALIZER;

/* dlopen and resolve everything. Idempotent; safe to call from multiple
   threads. Called from CreateSwapchain (which is past the loader-mutex
   window that causes the deadlock if we'd been DT_NEEDED-linked). */
static bool lib_load(void) {
    pthread_mutex_lock(&g_libs_lock);
    if (g_libs.loaded) {
        pthread_mutex_unlock(&g_libs_lock);
        return true;
    }
    /* Open libraries. libX11 first (libGL pulls it transitively, but we want
       to control which version). RTLD_GLOBAL so GLX dispatch resolves any
       symbols it needs across the libraries. */
    g_libs.handle_x11 = dlopen("libX11.so.6", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_libs.handle_x11) g_libs.handle_x11 = dlopen("libX11.so",   RTLD_LAZY | RTLD_GLOBAL);
    g_libs.handle_gl  = dlopen("libGL.so.1",  RTLD_LAZY | RTLD_GLOBAL);
    if (!g_libs.handle_gl)  g_libs.handle_gl  = dlopen("libGL.so",    RTLD_LAZY | RTLD_GLOBAL);
    g_libs.handle_glx = dlopen("libGLX.so.0", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_libs.handle_glx) g_libs.handle_glx = g_libs.handle_gl;  /* libGL provides glX* on most stacks */

    if (!g_libs.handle_x11 || !g_libs.handle_gl) {
        fprintf(stderr, "[" "VK_LAYER_TEGRA_x11_present" "] lib_load: failed to dlopen libX11/libGL (%s)\n",
                dlerror());
        pthread_mutex_unlock(&g_libs_lock);
        return false;
    }

    /* Resolve each symbol. We try the GL handle first for GL/GLX, then GLX
       handle as fallback, then global. For X11 symbols, X11 handle. */
#define RESOLVE_X11(name, ret, args) \
    g_libs.name = (PFN_##name)dlsym(g_libs.handle_x11, #name); \
    if (!g_libs.name) g_libs.name = (PFN_##name)dlsym(RTLD_DEFAULT, #name);
#define RESOLVE_GL(name, ret, args) \
    g_libs.name = (PFN_##name)dlsym(g_libs.handle_gl, #name); \
    if (!g_libs.name) g_libs.name = (PFN_##name)dlsym(RTLD_DEFAULT, #name);
#define RESOLVE_GLX(name, ret, args) \
    g_libs.name = (PFN_##name)dlsym(g_libs.handle_glx, #name); \
    if (!g_libs.name) g_libs.name = (PFN_##name)dlsym(g_libs.handle_gl, #name); \
    if (!g_libs.name) g_libs.name = (PFN_##name)dlsym(RTLD_DEFAULT, #name);
    X11_FUNCS(RESOLVE_X11)
    GL_FUNCS (RESOLVE_GL)
    GLX_FUNCS(RESOLVE_GLX)
#undef RESOLVE_X11
#undef RESOLVE_GL
#undef RESOLVE_GLX

    /* Now that XInitThreads is resolved, call it before any other Xlib
       function fires. Xlib requires this to enable its internal locking
       when multiple threads use Xlib (our worker thread has its own
       Display, but Xlib's internal locking is still a process-wide thing).
       Has to be called before any other Xlib call in the process. */
    if (g_libs.XInitThreads) g_libs.XInitThreads();

    g_libs.loaded = true;
    pthread_mutex_unlock(&g_libs_lock);
    return true;
}

/* Source-level redirection: every callsite below that says glXXX(...) or
   XXX(...) for X11 functions becomes g_libs.glXXX(...) without source
   changes. */
#define DECL_REMAP(name, ret, args) static const PFN_##name name##_indirect = NULL; (void)name##_indirect;
/* The above is unused; what we really want is a per-name #define. */
#undef DECL_REMAP

#define REMAP(name, ret, args) static inline ret name args;
/* Also unused — we just use the literal macros below. */
#undef REMAP

#define XInitThreads              (g_libs.XInitThreads)
#define XOpenDisplay              (g_libs.XOpenDisplay)
#define XCloseDisplay             (g_libs.XCloseDisplay)
#define XInternAtom               (g_libs.XInternAtom)
#define XCreateColormap           (g_libs.XCreateColormap)
#define XCreateWindow             (g_libs.XCreateWindow)
#define XDestroyWindow            (g_libs.XDestroyWindow)
#define XFreeColormap             (g_libs.XFreeColormap)
#define XFree                     (g_libs.XFree)
#define XMapWindow                (g_libs.XMapWindow)
#define XGetGeometry              (g_libs.XGetGeometry)
#define XGetWindowAttributes      (g_libs.XGetWindowAttributes)
#define XResizeWindow             (g_libs.XResizeWindow)
#define XFlush                    (g_libs.XFlush)
#define XChangeProperty           (g_libs.XChangeProperty)

#define glGetError                (g_libs.glGetError)
#define glClear                   (g_libs.glClear)
#define glClearColor              (g_libs.glClearColor)
#define glViewport                (g_libs.glViewport)
#define glActiveTexture           (g_libs.glActiveTexture)
#define glBindTexture             (g_libs.glBindTexture)
#define glGenTextures             (g_libs.glGenTextures)
#define glDeleteTextures          (g_libs.glDeleteTextures)
#define glTexParameteri           (g_libs.glTexParameteri)
#define glDrawArrays              (g_libs.glDrawArrays)
#define glBindVertexArray         (g_libs.glBindVertexArray)
#define glGenVertexArrays         (g_libs.glGenVertexArrays)
#define glDeleteVertexArrays      (g_libs.glDeleteVertexArrays)
#define glBindBuffer              (g_libs.glBindBuffer)
#define glGenBuffers              (g_libs.glGenBuffers)
#define glDeleteBuffers           (g_libs.glDeleteBuffers)
#define glBufferData              (g_libs.glBufferData)
#define glEnableVertexAttribArray (g_libs.glEnableVertexAttribArray)
#define glVertexAttribPointer     (g_libs.glVertexAttribPointer)
#define glCreateShader            (g_libs.glCreateShader)
#define glDeleteShader            (g_libs.glDeleteShader)
#define glShaderSource            (g_libs.glShaderSource)
#define glCompileShader           (g_libs.glCompileShader)
#define glGetShaderiv             (g_libs.glGetShaderiv)
#define glGetShaderInfoLog        (g_libs.glGetShaderInfoLog)
#define glCreateProgram           (g_libs.glCreateProgram)
#define glDeleteProgram           (g_libs.glDeleteProgram)
#define glAttachShader            (g_libs.glAttachShader)
#define glLinkProgram             (g_libs.glLinkProgram)
#define glGetProgramiv            (g_libs.glGetProgramiv)
#define glGetProgramInfoLog       (g_libs.glGetProgramInfoLog)
#define glUseProgram              (g_libs.glUseProgram)
#define glGetUniformLocation      (g_libs.glGetUniformLocation)
#define glUniform1i               (g_libs.glUniform1i)
#define glFlush                   (g_libs.glFlush)
#define glFinish                  (g_libs.glFinish)

#define glXChooseFBConfig         (g_libs.glXChooseFBConfig)
#define glXGetVisualFromFBConfig  (g_libs.glXGetVisualFromFBConfig)
#define glXCreateNewContext       (g_libs.glXCreateNewContext)
#define glXDestroyContext         (g_libs.glXDestroyContext)
#define glXMakeCurrent            (g_libs.glXMakeCurrent)
#define glXSwapBuffers            (g_libs.glXSwapBuffers)
#define glXGetProcAddressARB      (g_libs.glXGetProcAddressARB)

/* The vendored vk_layer.h used to define VK_LAYER_EXPORT for us, but the
   modern Vulkan-Loader removed it in favor of the layer's own visibility
   control. Define it portably here. With -fvisibility=hidden in the
   Makefile, this attribute is what makes vkGetInstanceProcAddr,
   vkGetDeviceProcAddr, and vkNegotiateLoaderLayerInterfaceVersion the
   only externally visible symbols. */
#if defined(__GNUC__) && (__GNUC__ >= 4)
#  define VK_LAYER_EXPORT __attribute__((visibility("default")))
#else
#  define VK_LAYER_EXPORT
#endif

/* ----------------------------------------------------------------------- */
/* Configuration                                                           */
/* ----------------------------------------------------------------------- */

#define LAYER_NAME            "VK_LAYER_TEGRA_x11_present"
#define LAYER_VERSION         2
#define LAYER_DESC            "Tegra L4T r32.x Vulkan→GL X11 present relay"

/* Clamp image counts to a sane range. */
#define MIN_IMAGES   2
#define MAX_IMAGES   8

/* Default image count if the app requests something outside the range. */
#define DEFAULT_IMAGES 3

/* ----------------------------------------------------------------------- */
/* Logging                                                                 */
/* ----------------------------------------------------------------------- */

static int g_log_level = 1;   /* 0=silent, 1=warn/err, 2=info, 3=debug */
static FILE *g_log_fp = NULL;

/* Diagnostic mode: if VK_TEGRA_X11_PRESENT_DIAG=1 is set, every QueueSubmit
   is followed by DeviceWaitIdle. This catches GPU faults at the offending
   submit instead of letting them propagate to a later submit returning
   DEVICE_LOST. It's catastrophically slow (every submit becomes synchronous)
   and is only meant for one-shot fault localization. */
static bool g_diag_wait_after_submit = false;

static void layer_log_init(void) {
    /* X11 threading initialization is deferred until lib_load() runs at
       CreateSwapchain time. We don't link libX11 directly anymore (see the
       big comment near the top of this file for the loader-deadlock
       rationale), so we can't call XInitThreads here. */

    const char *lvl = getenv("VK_TEGRA_X11_PRESENT_LOG");
    if (lvl) g_log_level = atoi(lvl);
    const char *diag = getenv("VK_TEGRA_X11_PRESENT_DIAG");
    if (diag && atoi(diag) == 1) {
        g_diag_wait_after_submit = true;
        fprintf(stderr, "[" LAYER_NAME "] DIAG MODE: DeviceWaitIdle after every submit (SLOW)\n");
    }
    const char *path = getenv("VK_TEGRA_X11_PRESENT_LOG_FILE");
    if (path) {
        g_log_fp = fopen(path, "a");
        if (g_log_fp) setvbuf(g_log_fp, NULL, _IONBF, 0);
    }
    /* Make stderr unbuffered so log lines survive abnormal exit. */
    setvbuf(stderr, NULL, _IONBF, 0);
}

static void layer_logv(int lvl, const char *prefix, const char *fmt, va_list ap) {
    if (lvl > g_log_level) return;
    char buf[2048];
    int n = snprintf(buf, sizeof(buf), "[" LAYER_NAME " %s] ", prefix);
    if (n > 0 && n < (int)sizeof(buf))
        vsnprintf(buf + n, sizeof(buf) - n, fmt, ap);
    fputs(buf, stderr); fputc('\n', stderr);
    if (g_log_fp) { fputs(buf, g_log_fp); fputc('\n', g_log_fp); }
}

#define LOG_ERR(fmt, ...)   do { if (g_log_level >= 1) layer_log(1, "ERR",  fmt, ##__VA_ARGS__); } while (0)
#define LOG_WARN(fmt, ...)  do { if (g_log_level >= 1) layer_log(1, "WARN", fmt, ##__VA_ARGS__); } while (0)
#define LOG_INFO(fmt, ...)  do { if (g_log_level >= 2) layer_log(2, "info", fmt, ##__VA_ARGS__); } while (0)
#define LOG_DBG(fmt, ...)   do { if (g_log_level >= 3) layer_log(3, "dbg ", fmt, ##__VA_ARGS__); } while (0)

static void layer_log(int lvl, const char *prefix, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); layer_logv(lvl, prefix, fmt, ap); va_end(ap);
}

/* Log every non-success Vulkan result from a call our layer makes. */
#define VK_CHECK(call) ({ \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) layer_log(1, "VK", "%s:%d %s -> %d", __func__, __LINE__, #call, _r); \
    _r; \
})

/* ----------------------------------------------------------------------- */
/* Layer enable / disable                                                  */
/* ----------------------------------------------------------------------- */

static bool g_layer_disabled = false;

static void layer_check_disabled(void) {
    const char *d = getenv("VK_TEGRA_X11_PRESENT_DISABLE");
    if (d && d[0] == '1') {
        g_layer_disabled = true;
        LOG_INFO("layer disabled via VK_TEGRA_X11_PRESENT_DISABLE=1 (passthrough)");
    }
}

/* ----------------------------------------------------------------------- */
/* Dispatch tables                                                         */
/* ----------------------------------------------------------------------- */

typedef struct {
    /* From next layer / loader */
    PFN_vkGetInstanceProcAddr               GetInstanceProcAddr;
    PFN_vkDestroyInstance                   DestroyInstance;
    PFN_vkEnumeratePhysicalDevices          EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties       GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;

    /* WSI bits we override or pass through */
    PFN_vkCreateXlibSurfaceKHR              CreateXlibSurfaceKHR;
    PFN_vkCreateXcbSurfaceKHR               CreateXcbSurfaceKHR;
    PFN_vkDestroySurfaceKHR                 DestroySurfaceKHR;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR        GetPhysicalDeviceSurfaceSupportKHR;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR   GetPhysicalDeviceSurfaceCapabilitiesKHR;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR        GetPhysicalDeviceSurfaceFormatsKHR;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR   GetPhysicalDeviceSurfacePresentModesKHR;

    /* VK_KHR_get_surface_capabilities2 — newer "2" variants that take an
       extensible chain instead of plain output structs. PPSSPP and several
       other modern Vulkan apps prefer these. We have to intercept them too;
       otherwise the underlying driver answers based on its own (broken) WSI
       state, and the app gets a surface capabilities object that doesn't
       match what we report for the v1 variants. The result is the app
       configuring its swapchain for one set of capabilities and then trying
       to use it against a different set — typically a NULL deref on the
       framebuffer/image-view chain that depends on the mismatched format. */
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR  GetPhysicalDeviceSurfaceCapabilities2KHR;
    PFN_vkGetPhysicalDeviceSurfaceFormats2KHR       GetPhysicalDeviceSurfaceFormats2KHR;

    PFN_vkGetPhysicalDeviceImageFormatProperties2 GetPhysicalDeviceImageFormatProperties2;
    PFN_vkGetPhysicalDeviceExternalSemaphoreProperties GetPhysicalDeviceExternalSemaphoreProperties;
} InstanceDispatch;

typedef struct {
    PFN_vkGetDeviceProcAddr      GetDeviceProcAddr;
    PFN_vkDestroyDevice          DestroyDevice;
    PFN_vkDeviceWaitIdle         DeviceWaitIdle;
    PFN_vkGetDeviceQueue         GetDeviceQueue;
    PFN_vkQueueSubmit            QueueSubmit;
    PFN_vkQueueWaitIdle          QueueWaitIdle;

    PFN_vkCreateImage            CreateImage;
    PFN_vkDestroyImage           DestroyImage;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkAllocateMemory         AllocateMemory;
    PFN_vkFreeMemory             FreeMemory;
    PFN_vkBindImageMemory        BindImageMemory;
    PFN_vkCreateSemaphore        CreateSemaphore;
    PFN_vkDestroySemaphore       DestroySemaphore;
    PFN_vkCreateFence            CreateFence;
    PFN_vkDestroyFence           DestroyFence;
    PFN_vkResetFences            ResetFences;
    PFN_vkWaitForFences          WaitForFences;
    PFN_vkGetFenceStatus         GetFenceStatus;

    PFN_vkGetMemoryFdKHR         GetMemoryFdKHR;
    PFN_vkGetSemaphoreFdKHR      GetSemaphoreFdKHR;
    PFN_vkCmdPipelineBarrier     CmdPipelineBarrier;
    PFN_vkEndCommandBuffer       EndCommandBuffer;
    PFN_vkCreateRenderPass       CreateRenderPass;
    /* v2 variants — core in Vulkan 1.2, also available as KHR extension.
       Vulkan 1.2 apps (e.g. Play PS2 emulator) use these instead of v1. */
    PFN_vkCreateRenderPass2      CreateRenderPass2;
    PFN_vkCreateRenderPass2      CreateRenderPass2KHR;

    /* The real WSI calls (we delegate format/presentmode queries to them
       sometimes, but otherwise we replace these completely). */
    PFN_vkCreateSwapchainKHR     CreateSwapchainKHR;
    PFN_vkDestroySwapchainKHR    DestroySwapchainKHR;
    PFN_vkGetSwapchainImagesKHR  GetSwapchainImagesKHR;
    PFN_vkAcquireNextImageKHR    AcquireNextImageKHR;
    PFN_vkQueuePresentKHR        QueuePresentKHR;
} DeviceDispatch;

/* ----------------------------------------------------------------------- */
/* Per-instance and per-device state, looked up by dispatch key            */
/* ----------------------------------------------------------------------- */

#define HASH_BUCKETS 64

typedef struct InstNode {
    void *key;
    VkInstance instance;
    InstanceDispatch d;
    bool external_mem_caps;
    bool external_sem_caps;
    /* True when no NVIDIA physical device was found at CreateInstance time.
       All surface creation and WSI calls for this instance pass through to
       the ICD unchanged so the layer is fully transparent for non-NVIDIA
       instances (llvmpipe, Mesa, etc.). */
    bool passthrough;
    struct InstNode *next;
} InstNode;
static InstNode *g_inst_table[HASH_BUCKETS];
static pthread_mutex_t g_inst_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct DevNode {
    void *key;
    VkDevice device;
    VkPhysicalDevice phys;
    /* When true this is a non-NVIDIA device.  d.GetDeviceProcAddr holds
       next_gdpa so layer_GetDeviceProcAddr can return the next layer's
       functions, making the layer fully transparent for this device.
       Only CreateSwapchainKHR and DestroyDevice are still intercepted to
       handle cleanup of state created by layer_CreateXlibSurfaceKHR /
       layer_CreateXcbSurfaceKHR (the icd_surface in our Surface*). */
    bool passthrough;
    VkInstance inst;
    DeviceDispatch d;
    InstanceDispatch *idisp;     /* points into the InstNode for this device */
    VkPhysicalDeviceMemoryProperties memp;
    uint32_t graphics_qfi;
    VkQueue graphics_queue;      /* lazy-resolved */
    /* Queue submit serialization.

       VkQueue is "externally synchronized" — per the Vulkan spec, only one
       thread may call vkQueueSubmit on a given queue at a time. The app is
       responsible for that mutex, but our layer's own bridge submits in
       Acquire/Present also touch the queue. If the app has a separate render
       thread (PPSSPP does) that submits concurrently with our bridge, the
       Nvidia Tegra driver hits an internal race and the GPU faults — observed
       as DEVICE_LOST on subsequent submits.

       We serialize on a per-device mutex covering ALL submits — ours and the
       app's (through our layer_QueueSubmit wrapper). The app may have its
       own mutex too; that's fine, double-locking a single submit costs only
       a few ns. The point is that no submit from any source happens
       concurrently with another. */
    pthread_mutex_t submit_lock;
    int             submit_inflight;  /* DEBUG: should always be 0 or 1 */
    struct DevNode *next;
} DevNode;
static DevNode *g_dev_table[HASH_BUCKETS];
static pthread_mutex_t g_dev_lock = PTHREAD_MUTEX_INITIALIZER;

/* The Vulkan loader's "dispatch key" is the first sizeof(void*) bytes of
   the object — both VkInstance and VkDevice are dispatchable handles. */
/* The Vulkan loader's "dispatch key" is the first sizeof(void*) bytes of
   the dispatchable object — it points to the loader's per-{instance,device}
   dispatch table. ALL dispatchable handles derived from the same
   device share that pointer (the device, every VkQueue, every VkCommandBuffer
   from that device). We use it as a hash key so any of those handles can
   resolve back to our DevNode.

   Important: pass the handle itself, NOT &handle. The argument is a
   dispatchable handle (a pointer-typed value). We dereference it to read
   the first 8 bytes. */
static void *dispatch_key(const void *handle) {
    return *(void * const *)handle;
}

static unsigned bucket(void *k) {
    uintptr_t x = (uintptr_t)k;
    x ^= x >> 16; x *= 0x9E3779B1u; x ^= x >> 16;
    return (unsigned)(x & (HASH_BUCKETS - 1));
}

static InstNode *inst_lookup(void *key) {
    pthread_mutex_lock(&g_inst_lock);
    InstNode *n = g_inst_table[bucket(key)];
    while (n && n->key != key) n = n->next;
    pthread_mutex_unlock(&g_inst_lock);
    return n;
}
static void inst_insert(InstNode *n) {
    pthread_mutex_lock(&g_inst_lock);
    unsigned b = bucket(n->key); n->next = g_inst_table[b]; g_inst_table[b] = n;
    pthread_mutex_unlock(&g_inst_lock);
}
static void inst_remove(void *key) {
    pthread_mutex_lock(&g_inst_lock);
    unsigned b = bucket(key);
    InstNode **p = &g_inst_table[b];
    while (*p && (*p)->key != key) p = &(*p)->next;
    if (*p) { InstNode *n = *p; *p = n->next; free(n); }
    pthread_mutex_unlock(&g_inst_lock);
}

static DevNode *dev_lookup(void *key) {
    pthread_mutex_lock(&g_dev_lock);
    DevNode *n = g_dev_table[bucket(key)];
    while (n && n->key != key) n = n->next;
    pthread_mutex_unlock(&g_dev_lock);
    return n;
}
static void dev_insert(DevNode *n) {
    pthread_mutex_lock(&g_dev_lock);
    unsigned b = bucket(n->key); n->next = g_dev_table[b]; g_dev_table[b] = n;
    pthread_mutex_unlock(&g_dev_lock);
}
static void dev_remove(void *key) {
    pthread_mutex_lock(&g_dev_lock);
    unsigned b = bucket(key);
    DevNode **p = &g_dev_table[b];
    while (*p && (*p)->key != key) p = &(*p)->next;
    if (*p) { DevNode *n = *p; *p = n->next; free(n); }
    pthread_mutex_unlock(&g_dev_lock);
}

/* For queue->device lookup. Vulkan queues are dispatchable; their key is the
   parent device's. */

/* Serialized queue submit. Acquires dev->submit_lock, calls the driver's
   QueueSubmit, releases. This is the ONLY way our layer touches the queue —
   the app's submits go through layer_QueueSubmit which uses the same lock.

   The Vulkan spec requires external synchronization on the queue parameter
   to vkQueueSubmit: the application must ensure only one thread submits to
   a given queue at a time. We have to provide that synchronization for
   our own bridge submits AND for the app's submits going through our
   wrapper, because the app's render thread and our Acquire/Present can
   both end up on the same VkQueue. */
static VkResult queue_submit_locked(DevNode *dev, VkQueue queue,
                                    uint32_t submitCount,
                                    const VkSubmitInfo *pSubmits,
                                    VkFence fence) {
    pthread_mutex_lock(&dev->submit_lock);
#ifndef NDEBUG
    /* Sanity check: with the mutex held, exactly one thread should be in
       the critical section at a time. If this ever exceeds 1 the lock is
       broken. Production builds skip this check. */
    int n = __atomic_add_fetch(&dev->submit_inflight, 1, __ATOMIC_SEQ_CST);
    if (n != 1) {
        LOG_ERR("queue_submit_locked: %d threads inside lock simultaneously! "
                "(self=%lu queue=%p)",
                n, (unsigned long)pthread_self(), (void*)queue);
    }
#endif
    VkResult r = dev->d.QueueSubmit(queue, submitCount, pSubmits, fence);
#ifndef NDEBUG
    __atomic_sub_fetch(&dev->submit_inflight, 1, __ATOMIC_SEQ_CST);
#endif
    pthread_mutex_unlock(&dev->submit_lock);
    return r;
}

/* ----------------------------------------------------------------------- */
/* Surfaces                                                                */
/* ----------------------------------------------------------------------- */

typedef enum {
    SURF_XLIB,
    SURF_XCB,
} SurfaceKind;

typedef struct Surface {
    /* Magic value so we can recognize our own VkSurfaceKHR-shaped handles. */
    uint64_t magic;
    SurfaceKind kind;
    Display *dpy;            /* Xlib handle. For XCB-only apps, we open one ourselves. */
    bool owns_dpy;
    Window window;
    /* Real ICD-owned VkSurfaceKHR created alongside our wrapper.  Used
       when a non-NVIDIA device needs a valid ICD surface handle — the ICD
       does not know about our Surface* pointer.  Destroyed in
       DestroySurfaceKHR together with our wrapper. */
    VkSurfaceKHR icd_surface;
    VkInstance   icd_inst;
    /* The GLX context lives in the SwapchainData, not here, because it
       must match the GLXFBConfig of the rendering format and the app
       chooses format at swapchain creation time. */
} Surface;

#define SURFACE_MAGIC 0x53524654594c5253ULL  /* "SRFTYLSR" backwards-ish */

/* Cast a VkSurfaceKHR (non-dispatchable, 64-bit) into our Surface*. */
static Surface *as_surface(VkSurfaceKHR s) {
    Surface *p = (Surface *)(uintptr_t)s;
    if (!p || p->magic != SURFACE_MAGIC) return NULL;
    return p;
}

/* ----------------------------------------------------------------------- */
/* Swapchains                                                              */
/* ----------------------------------------------------------------------- */

typedef struct PerImage {
    VkImage          image;        /* Vulkan image, OPAQUE_FD exportable */
    VkDeviceMemory   memory;
    VkDeviceSize     size;
    int              mem_fd;       /* fd from vkGetMemoryFdKHR (kept open) */

    VkSemaphore      vk_render_done;  /* Vulkan signals, GL waits.
                                         Exported as OPAQUE_FD. */
    VkSemaphore      gl_sample_done;  /* GL signals, Vulkan waits at next acquire.
                                         Exported as OPAQUE_FD. */
    int              vk_render_done_fd;
    int              gl_sample_done_fd;

    GLuint           gl_memobj;       /* glImportMemoryFdEXT */
    GLuint           gl_texture;      /* bound to gl_memobj */
    GLuint           gl_render_sem;   /* glImportSemaphoreFdEXT of vk_render_done */
    GLuint           gl_sample_sem;   /* glImportSemaphoreFdEXT of gl_sample_done */

    /* Per-image fence: used by vkAcquireNextImageKHR to provide CPU-side
       backpressure. Without this, Acquire returns immediately and the app's
       render loop is unblocked by the layer, causing it to spin at maximum
       framerate while the worker is the one waiting on vsync. With it,
       Acquire blocks on the fence until our bridge submit (which itself
       waits on gl_sample_done) completes — same semantic as real WSI
       Acquire which blocks until a swapchain image is genuinely free. */
    VkFence          acquire_fence;

    /* Tracking the acquire state.

       Vulkan's swapchain model says an acquired image is either in the
       app's possession or in the presentation engine. We treat it as:
         - "acquired by app" between Acquire and Present
         - "in flight in GL" between Present and the moment glXSwapBuffers
           returns
         - "free" after glXSwapBuffers returns

       We use gl_sample_done as the gate: when the app calls Acquire and
       requests image idx, we make the app's acquire semaphore wait on
       gl_sample_done[idx] via a bridge submit. */
    bool             acquired;        /* currently held by app */
    bool             in_flight;       /* GL has work pending on it */
} PerImage;

typedef struct Swapchain {
    uint64_t magic;
    DevNode *dev;
    Surface *surf;

    /* Properties */
    uint32_t      image_count;
    VkExtent2D    extent;
    VkFormat      format;
    VkColorSpaceKHR color_space;
    VkPresentModeKHR present_mode;
    /* Usage flags the app requested for swapchain images. We honor these
       (OR'd with what we need ourselves) when creating our images. PPSSPP
       in particular asks for INPUT_ATTACHMENT_BIT for its subpass effects;
       if we don't provide it, the app's later render passes / pipelines
       are valid at creation time but the GPU faults on use. */
    VkImageUsageFlags image_usage;

    PerImage      images[MAX_IMAGES];

    /* X / GLX resources owned by this swapchain.

       GLX requires the drawable's visual to match the GLX context's
       FBConfig visual. The application's surface window was created
       without GLX in mind and almost certainly has a visual that no GLX
       FBConfig matches — glXMakeCurrent on it returns BadMatch. We work
       around this by creating a child X window inside the app's window
       with our own chosen visual, sized to match the parent, and using
       that as the GLX drawable. The app never sees it; the X server
       composites it into the parent's area automatically. */
    GLXFBConfig   fbcfg;
    XVisualInfo  *visinfo;
    Colormap      child_colormap;
    Window        child_window;     /* the actual GLX drawable */
    GLXContext    glctx;
    bool          glctx_owned;
    /* The parent X window's actual size, refreshed each present from
       XGetGeometry. We resize the child window to match when it changes. */
    int           win_w, win_h;

    /* Worker's own X Display connection.

       Xlib is not thread-safe to share a single Display* across threads
       even with XInitThreads — the internal sequencer asserts if two
       threads make X requests concurrently. Our main thread uses
       sc->surf->dpy (the surface's Display) for swapchain setup and
       teardown; the worker thread uses its own dedicated connection
       (worker_dpy). The Window XID is a server-side ID and can be
       safely referenced from either connection. The GLX context is also
       a server-side object and can be made-current on the worker's
       connection. */
    Display      *worker_dpy;

    /* GL program state for the textured-quad blit */
    GLuint        prog;
    GLuint        vao;
    GLuint        vbo;
    GLint         u_tex;

    /* Cached extension entrypoints, resolved once per swapchain when its
       GLX context is first made current. */
    PFNGLXSWAPINTERVALEXTPROC        glXSwapIntervalEXT;
    PFNGLCREATEMEMORYOBJECTSEXTPROC  glCreateMemoryObjectsEXT;
    PFNGLIMPORTMEMORYFDEXTPROC       glImportMemoryFdEXT;
    PFNGLTEXSTORAGEMEM2DEXTPROC      glTexStorageMem2DEXT;
    PFNGLDELETEMEMORYOBJECTSEXTPROC  glDeleteMemoryObjectsEXT;
    PFNGLGENSEMAPHORESEXTPROC        glGenSemaphoresEXT;
    PFNGLIMPORTSEMAPHOREFDEXTPROC    glImportSemaphoreFdEXT;
    PFNGLWAITSEMAPHOREEXTPROC        glWaitSemaphoreEXT;
    PFNGLSIGNALSEMAPHOREEXTPROC      glSignalSemaphoreEXT;
    PFNGLDELETESEMAPHORESEXTPROC     glDeleteSemaphoresEXT;

    /* GLX_SGI_video_sync entrypoints. When available, the worker uses
       glXWaitVideoSyncSGI to block on the actual hardware vblank instead
       of relying on glXSwapBuffers to do so. NVIDIA's Tegra L4T r32.x
       implements glXSwapBuffers's vsync wait as a sched_yield() spin loop
       that pins the worker thread at ~100% CPU. glXWaitVideoSyncSGI does
       a real kernel-side vblank wait (DRM_IOCTL_WAIT_VBLANK or similar),
       so the thread actually sleeps. Pattern lifted from KWin's
       SGIVideoSyncVsyncMonitor — see
       plugins/platforms/x11/standalone/glxbackend.cpp in the KWin tree. */
    int  (*glXGetVideoSyncSGI )(unsigned int *count);
    int  (*glXWaitVideoSyncSGI)(int divisor, int remainder, unsigned int *count);

    /* Acquire ring */
    uint32_t      next_acquire;       /* round-robin starting point */

    /* Per-swapchain command pool for our bridge submits. */
    VkCommandPool  cpool;

    /* Async GL worker.

       Architecture: a dedicated thread owns the GLX context — made current
       once at thread start, never un-made until shutdown. This avoids the
       per-present cost of glXMakeCurrent and decouples the app's render
       thread from glXSwapBuffers's vsync wait. On Nvidia, glXSwapBuffers
       with swap interval >= 1 spin-waits on the CPU side until vblank;
       running it on a dedicated thread means it doesn't burn the app's
       core. The app thread submits a job and returns immediately.

       Pending slot: a single image-index awaiting present, plus a flag.
       This single-slot design is the backpressure mechanism — if the
       worker hasn't finished the last present when the app calls Present
       again, the app blocks in the post until the slot frees. With NIMG=3
       images and triple-buffered render, normal usage stays fully
       pipelined. */
    pthread_t        worker;
    bool             worker_running;
    pthread_mutex_t  worker_lock;
    pthread_cond_t   worker_cv_pending;   /* worker waits on this when idle */
    pthread_cond_t   worker_cv_done;      /* app waits on this when posting to full slot */
    bool             worker_pending;      /* slot has work? */
    uint32_t         worker_pending_idx;  /* image index in pending slot */
    /* VK_GOOGLE_display_timing: presentID and desiredPresentTime supplied
       by the app for the pending work (zero if not set). The worker
       captures actualPresentTime when the SGI vblank wait returns and
       pushes a history entry. */
    uint32_t         worker_pending_present_id;
    uint64_t         worker_pending_desired_ns;
    bool             worker_quit;

    /* Display-timing history ring, written by the worker after each
       vblank, read by vkGetPastPresentationTimingGOOGLE. The lock
       protects head/tail/count; the contents themselves are pure
       value-copies so no aliasing concern.

       64 entries is generous — the spec just requires us to remember
       "the most recent" presents; apps that drain regularly never see
       it fill up. If full, the oldest entry is overwritten — apps that
       don't poll lose old history but always see recent. */
    pthread_mutex_t  timing_lock;
    uint32_t         timing_count;        /* number of valid entries */
    uint32_t         timing_head;         /* index of oldest entry */
    VkPastPresentationTimingGOOGLE timing_ring[64];
    /* Measured refresh duration in nanoseconds. Filled in at swapchain
       create time from one observed inter-vblank interval, falling back
       to a conservative 60 Hz if measurement fails. */
    uint64_t         refresh_duration_ns;

    pthread_mutex_t lock;
} Swapchain;

#define SWAPCHAIN_MAGIC 0x5357415043484149ULL    /* "SWAPCHAI" — 8 bytes, fits uint64_t */

static Swapchain *as_swapchain(VkSwapchainKHR s) {
    Swapchain *p = (Swapchain *)(uintptr_t)s;
    if (!p || p->magic != SWAPCHAIN_MAGIC) return NULL;
    return p;
}

static void track_swapchain(Swapchain *sc);
static void untrack_swapchain(Swapchain *sc);

/* ----------------------------------------------------------------------- */
/* GLX helpers                                                             */
/* ----------------------------------------------------------------------- */

/* Choose an FBConfig matching the requested swapchain format. We always
   double-buffer; depth is irrelevant since GL renders only a textured quad.
   We need an FBConfig whose visual matches the X window's visual, but
   that's a problem the app already solved at window creation time — the
   window has SOME visual, and we ask GLX to find a doublebuffered RGBA
   FBConfig of equivalent depth. */
/* Pick a doublebuffered RGBA GLX framebuffer config. We prefer one whose
   X visual depth is 32 — that's the ARGB visual that compositors will
   alpha-blend with the desktop. With a 24-bit visual the X server treats
   the window as opaque regardless of how much alpha we render into it,
   which manifests as black borders around apps that use CSD shadows
   (Chromium, GTK3/4 client-side-decorated apps, Electron, etc.). If a
   32-bit visual isn't available we accept any RGBA FBConfig and let the
   compositor render the window opaquely.

   parent_depth, if nonzero, is a preference toward matching the X parent
   window's visual depth — if the app's window is depth 24 there's no
   compositor blending to preserve and the simpler/faster 24-bit visual
   is fine. */
static GLXFBConfig pick_fbconfig(Display *dpy, int screen, VkFormat fmt,
                                 int parent_depth, XVisualInfo **out_vi) {
    int red=8, green=8, blue=8, alpha=8;
    /* SRGB needs GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB. */
    bool want_srgb = (fmt == VK_FORMAT_R8G8B8A8_SRGB || fmt == VK_FORMAT_B8G8R8A8_SRGB);

    int attrs[32]; int n = 0;
    attrs[n++] = GLX_X_RENDERABLE;     attrs[n++] = True;
    attrs[n++] = GLX_DRAWABLE_TYPE;    attrs[n++] = GLX_WINDOW_BIT;
    attrs[n++] = GLX_RENDER_TYPE;      attrs[n++] = GLX_RGBA_BIT;
    attrs[n++] = GLX_RED_SIZE;         attrs[n++] = red;
    attrs[n++] = GLX_GREEN_SIZE;       attrs[n++] = green;
    attrs[n++] = GLX_BLUE_SIZE;        attrs[n++] = blue;
    attrs[n++] = GLX_ALPHA_SIZE;       attrs[n++] = alpha;
    attrs[n++] = GLX_DEPTH_SIZE;       attrs[n++] = 0;
    attrs[n++] = GLX_DOUBLEBUFFER;     attrs[n++] = True;
    if (want_srgb) { attrs[n++] = GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB; attrs[n++] = True; }
    attrs[n++] = None;

    int nfb = 0;
    GLXFBConfig *fbs = glXChooseFBConfig(dpy, screen, attrs, &nfb);
    if (!fbs || nfb == 0) {
        /* Retry without SRGB if that was the only constraint failing. */
        if (want_srgb) {
            int j = 0;
            while (attrs[j] != GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB && attrs[j] != None) j++;
            if (attrs[j] != None) { attrs[j] = None; }
            fbs = glXChooseFBConfig(dpy, screen, attrs, &nfb);
        }
    }
    if (!fbs || nfb == 0) return NULL;

    /* Search returned FBConfigs for one whose visual matches the parent
       window's depth (if specified) or is depth 32 (preferred for
       compositor alpha blending). Fall back to the first FBConfig if no
       match. */
    GLXFBConfig pick = fbs[0];
    XVisualInfo *pick_vi = glXGetVisualFromFBConfig(dpy, pick);
    int want_depth = (parent_depth == 32 || parent_depth == 24) ? parent_depth : 32;
    for (int i = 0; i < nfb; i++) {
        XVisualInfo *vi = glXGetVisualFromFBConfig(dpy, fbs[i]);
        if (!vi) continue;
        if (vi->depth == want_depth) {
            if (pick_vi) XFree(pick_vi);
            pick = fbs[i];
            pick_vi = vi;
            break;
        }
        XFree(vi);
    }
    /* If nothing matched our preferred depth, fall back to ANY 32-bit
       visual — the compositor case is more important than matching the
       parent's depth exactly. */
    if (pick_vi && pick_vi->depth != want_depth && want_depth != 32) {
        for (int i = 0; i < nfb; i++) {
            XVisualInfo *vi = glXGetVisualFromFBConfig(dpy, fbs[i]);
            if (!vi) continue;
            if (vi->depth == 32) {
                XFree(pick_vi);
                pick = fbs[i];
                pick_vi = vi;
                break;
            }
            XFree(vi);
        }
    }
    *out_vi = pick_vi;
    XFree(fbs);
    return pick;
}

static GLXContext create_glx_context(Display *dpy, GLXFBConfig fbc) {
    PFNGLXCREATECONTEXTATTRIBSARBPROC f =
        (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddressARB((const GLubyte*)"glXCreateContextAttribsARB");
    if (f) {
        int a[] = {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
            GLX_CONTEXT_MINOR_VERSION_ARB, 3,
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            None
        };
        GLXContext c = f(dpy, fbc, NULL, True, a);
        if (c) return c;
    }
    return glXCreateNewContext(dpy, fbc, GLX_RGBA_TYPE, NULL, True);
}

/* Worker thread main loop. Owns the GLX context for the swapchain's lifetime;
   the context is made current once at thread start and never released until
   the thread exits. The thread waits on worker_cv_pending for work, processes
   one frame at a time, and signals worker_cv_done when the slot is free.

   The worker opens its OWN X Display* connection (sc->worker_dpy) and uses
   that for all X operations. Sharing an Xlib Display* across threads is
   unsafe even with XInitThreads — the internal sequencer asserts. By giving
   the worker its own connection, X requests from both threads are
   independent. The Window and GLX context are server-side objects and can
   be referenced from any connection. */
static void *worker_thread_main(void *arg) {
    Swapchain *sc = (Swapchain *)arg;

    /* Open the worker's own X connection. The Window XID we're going to
       render into is the same; only the connection differs. */
    sc->worker_dpy = XOpenDisplay(NULL);
    if (!sc->worker_dpy) {
        LOG_ERR("worker_thread_main: XOpenDisplay failed");
        pthread_mutex_lock(&sc->worker_lock);
        sc->worker_running = false;
        pthread_cond_broadcast(&sc->worker_cv_done);
        pthread_mutex_unlock(&sc->worker_lock);
        return NULL;
    }

    if (!glXMakeCurrent(sc->worker_dpy, sc->child_window, sc->glctx)) {
        LOG_ERR("worker_thread_main: glXMakeCurrent failed at startup");
        XCloseDisplay(sc->worker_dpy); sc->worker_dpy = NULL;
        pthread_mutex_lock(&sc->worker_lock);
        sc->worker_running = false;
        pthread_cond_broadcast(&sc->worker_cv_done);
        pthread_mutex_unlock(&sc->worker_lock);
        return NULL;
    }

    /* Swap interval: mirrors what any standard GL application uses.
       glXSwapBuffers(interval=1) is the correct vsync mechanism in both
       compositor and non-compositor environments:
         - Without compositor: the driver schedules an async DRM page flip
           and returns immediately; the display controller presents it at
           the next vblank.
         - With compositor: the compositor pipeline blocks glXSwapBuffers
           for ~1 vblank while it takes the buffer; its own rendering loop
           then presents at the next vblank.
       In both cases one call to glXSwapBuffers(interval=1) = one frame at
       the display's native refresh rate.  glXWaitVideoSyncSGI is a
       CPU-efficiency add-on layered on top — see the worker loop. */
    if (sc->glXSwapIntervalEXT) {
        int interval;
        if (sc->present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
            interval = 0;
        else if (sc->present_mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
            interval = -1;
        else
            interval = 1;   /* FIFO, MAILBOX */
        sc->glXSwapIntervalEXT(sc->worker_dpy, sc->child_window, interval);
    }

    int last_win_w = sc->win_w, last_win_h = sc->win_h;
    glViewport(0, 0, last_win_w, last_win_h);

    /* Tracks the previous vblank wall-clock time for refresh-duration
       measurement. Zero on the first iteration; set after each vsync
       wait. Local because the worker thread runs the same loop for the
       swapchain's lifetime. */
    uint64_t prev_vblank_ns = 0;
    /* EWMA of glXSwapBuffers duration, used to distinguish fast (async DRM
       flip, direct scanout) from slow (compositor blocking, ~1 vblank).
       Seeded with the first observed swap on the first iteration. */
    uint64_t avg_swap_ns = 0;
    bool avg_swap_valid = false;
    /* -1 = not yet known; 0 = SGI skipped (slow swap); 1 = SGI active (fast swap). */
    int prev_sgi_state = -1;
    /* Timestamp of the last periodic avg-swap log line (0 = not yet printed). */
    uint64_t last_swap_log_ns = 0;

    for (;;) {
        /* Wait for work or shutdown. */
        pthread_mutex_lock(&sc->worker_lock);
        while (!sc->worker_pending && !sc->worker_quit)
            pthread_cond_wait(&sc->worker_cv_pending, &sc->worker_lock);
        if (sc->worker_quit) {
            pthread_mutex_unlock(&sc->worker_lock);
            break;
        }
        uint32_t idx = sc->worker_pending_idx;
        uint32_t present_id = sc->worker_pending_present_id;
        uint64_t desired_ns = sc->worker_pending_desired_ns;
        pthread_mutex_unlock(&sc->worker_lock);

        /* Refresh window size if changed. Cheap when unchanged. Use the
           worker's own Display* for these X calls. */
        uint32_t ww = 0, wh = 0;
        Window root; int wx, wy; unsigned int wb, wd;
        if (XGetGeometry(sc->worker_dpy, sc->surf->window, &root, &wx, &wy, &ww, &wh, &wb, &wd)) {
            if ((int)ww != last_win_w || (int)wh != last_win_h) {
                XResizeWindow(sc->worker_dpy, sc->child_window, ww, wh);
                XFlush(sc->worker_dpy);
                last_win_w = (int)ww; last_win_h = (int)wh;
                glViewport(0, 0, last_win_w, last_win_h);
            }
        }

        /* GL: wait, sample, signal, swap. */
        GLenum srcLayouts[1] = { GL_LAYOUT_SHADER_READ_ONLY_EXT };
        sc->glWaitSemaphoreEXT(sc->images[idx].gl_render_sem, 0, NULL,
                               1, &sc->images[idx].gl_texture, srcLayouts);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sc->images[idx].gl_texture);
        glBindVertexArray(sc->vao);
        glUseProgram(sc->prog);
        glUniform1i(sc->u_tex, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        GLenum dstLayouts[1] = { GL_LAYOUT_SHADER_READ_ONLY_EXT };
        sc->glSignalSemaphoreEXT(sc->images[idx].gl_sample_sem, 0, NULL,
                                 1, &sc->images[idx].gl_texture, dstLayouts);
        /* glFinish forces all queued GL commands (including the sample draw
           above) to actually complete on the GPU before we proceed. Without
           it the draw may still be in flight when we reach glXSwapBuffers,
           and on the docked HDMI output this races with the display
           controller and manifests as a slowly-drifting tear line.
           glFinish closes that race at the cost of one CPU stall per frame —
           the stall is short because the sample draw is a single textured
           quad. */
        glFinish();

        /* Swap and adaptive SGI sleep.

           glXSwapBuffers(interval=1) provides correct vsync regardless of
           whether a compositor is running:
             - Without compositor: async DRM flip; returns immediately; the
               display controller presents at the next vblank.
             - With compositor: compositor pipeline blocks the call for ~1
               vblank while taking the buffer; the compositor presents at
               the next vblank.
           Either way, one swap = one frame at the display refresh rate.

           glXWaitVideoSyncSGI is a CPU-efficiency add-on: on NVIDIA Tegra
           the thread would otherwise spin (sched_yield) until the next
           work arrives from the app.  SGI puts it to sleep in the kernel
           for most of the inter-frame interval.

           The catch: SGI must only be called when the swap itself returned
           quickly.  If the compositor blocked the swap for ~1 vblank (i.e.
           swap_ns ≥ refresh_duration_ns/4), calling SGI afterwards would
           add a second vblank wait and cap the frame rate at 30Hz.  When
           the swap was already slow, the vblank is already consumed — skip
           SGI and return immediately to accept the next frame.

           This logic is entirely self-contained: it requires no knowledge
           of whether a compositor is running, and adapts correctly on the
           very next frame whenever the compositing state changes. */

        struct timespec _t0, _t1;
        clock_gettime(CLOCK_MONOTONIC, &_t0);
        glXSwapBuffers(sc->worker_dpy, sc->child_window);
        clock_gettime(CLOCK_MONOTONIC, &_t1);
        uint64_t swap_ns = ((uint64_t)_t1.tv_sec  * 1000000000ULL + _t1.tv_nsec)
                         - ((uint64_t)_t0.tv_sec  * 1000000000ULL + _t0.tv_nsec);
        XFlush(sc->worker_dpy);

        /* Update the EWMA of swap duration.  Seed with the first observed
           value to avoid a misleading zero on the first frame. */
        if (!avg_swap_valid) {
            avg_swap_ns   = swap_ns;
            avg_swap_valid = true;
        } else {
            /* 7/8 old + 1/8 new — window ≈ 8 frames (~130 ms at 60 Hz).
               Long enough to ride out 1-3 transient spikes (scheduler
               jitter, GPU hiccup, compositor bypass race on startup) but
               short enough to follow a real mode change within ~8 frames. */
            avg_swap_ns = (7 * avg_swap_ns + swap_ns) / 8;
        }

        /* SGI: only call when the average swap duration is consistently
           fast (vblank not yet consumed by the swap itself). */
        bool do_sgi = sc->glXWaitVideoSyncSGI
                      && sc->present_mode != VK_PRESENT_MODE_IMMEDIATE_KHR
                      && avg_swap_ns < sc->refresh_duration_ns / 4;

        if ((int)do_sgi != prev_sgi_state) {
            if (do_sgi)
                LOG_INFO("vsync path -> SGI kernel sleep "
                         "(avg swap %.2f ms, fast — direct scanout or bypass active)",
                         (double)avg_swap_ns / 1e6);
            else
                LOG_INFO("vsync path -> swap blocking "
                         "(avg swap %.2f ms >= %.2f ms threshold — compositor compositing window, SGI skipped)",
                         (double)avg_swap_ns / 1e6,
                         (double)(sc->refresh_duration_ns / 4) / 1e6);
            prev_sgi_state = (int)do_sgi;
        }

        if (do_sgi) {
            unsigned int count = 0;
            if (sc->glXGetVideoSyncSGI(&count) == 0)
                sc->glXWaitVideoSyncSGI(2, (count + 1) & 1, &count);
            else
                sc->glXWaitVideoSyncSGI(2, 0, &count);
        }

        /* Capture the vblank timestamp: after SGI if it ran (the SGI wait
           lands at the hardware vblank), or right after the slow swap if it
           didn't (the swap itself blocked until the compositor's vblank). */
        uint64_t actual_ns;
        {
            struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
            actual_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
        }

        /* Periodic debug log: avg swap duration once per second.
           Enable with VK_TEGRA_X11_PRESENT_LOG=2. */
        if (last_swap_log_ns == 0 || actual_ns - last_swap_log_ns >= 1000000000ULL) {
            LOG_INFO("avg swap %.2f ms  threshold %.2f ms  path: %s",
                    (double)avg_swap_ns / 1e6,
                    (double)(sc->refresh_duration_ns / 4) / 1e6,
                    do_sgi ? "SGI sleep" : "swap blocking");
            last_swap_log_ns = actual_ns;
        }

        /* Refine the reported refresh duration from observed
           inter-vblank intervals. EWMA over a long window keeps the
           value stable; bias initially toward the default until a few
           frames have elapsed. We deliberately use the inter-frame
           interval observed AT vsync, which means glXWaitVideoSyncSGI
           actually fired — outliers from missed vblanks would inflate
           the average. We filter to [13ms, 21ms] which covers
           50-75Hz; anything outside that range is treated as bogus and
           ignored (e.g. an app that hides the window briefly). */
        if (prev_vblank_ns != 0) {
            uint64_t dt = actual_ns - prev_vblank_ns;
            if (dt > 13000000ULL && dt < 21000000ULL) {
                uint64_t cur = sc->refresh_duration_ns;
                /* EWMA: 1/8 new, 7/8 old. Converges in ~30 frames from a
                   default to true rate, then tracks slow drift. */
                sc->refresh_duration_ns = (7 * cur + dt) / 8;
            }
        }
        prev_vblank_ns = actual_ns;

        /* If the app marked this present with a presentID (via
           VkPresentTimesInfoGOOGLE in pNext of VkPresentInfoKHR), push
           a history entry. Apps that don't use the extension never set
           presentID, so the zero check filters their non-tracked
           presents out — keeps the ring uncluttered for apps that do
           use both modes. */
        if (present_id != 0) {
            pthread_mutex_lock(&sc->timing_lock);
            uint32_t slot;
            if (sc->timing_count < 64) {
                slot = (sc->timing_head + sc->timing_count) & 63;
                sc->timing_count++;
            } else {
                /* Ring full — overwrite oldest, advance head. */
                slot = sc->timing_head;
                sc->timing_head = (sc->timing_head + 1) & 63;
            }
            VkPastPresentationTimingGOOGLE *e = &sc->timing_ring[slot];
            e->presentID          = present_id;
            e->desiredPresentTime = desired_ns;
            e->actualPresentTime  = actual_ns;
            e->earliestPresentTime = actual_ns;        /* we don't pre-empt */
            e->presentMargin       = 0;                 /* unknown / not tracked */
            pthread_mutex_unlock(&sc->timing_lock);
        }

        /* Mark slot free AFTER the present completes. This is the
           backpressure point: while worker_pending is true, any caller in
           worker_post blocks. */
        pthread_mutex_lock(&sc->worker_lock);
        sc->worker_pending = false;
        pthread_cond_broadcast(&sc->worker_cv_done);
        pthread_mutex_unlock(&sc->worker_lock);
    }

    glXMakeCurrent(sc->worker_dpy, None, NULL);
    XCloseDisplay(sc->worker_dpy);
    sc->worker_dpy = NULL;

    pthread_mutex_lock(&sc->worker_lock);
    sc->worker_running = false;
    pthread_cond_broadcast(&sc->worker_cv_done);
    pthread_mutex_unlock(&sc->worker_lock);
    return NULL;
}

/* Post an image index to the worker's pending slot.

   Behavior depends on the swapchain's present mode:

   - FIFO / FIFO_RELAXED (the default): blocks the caller if the slot is
     full. This is what gives us natural backpressure — the app's render
     loop is paced by the worker's swap rate. The function returns
     UINT32_MAX in *displaced_idx.

   - MAILBOX: replaces the slot contents without blocking. If the slot
     was already occupied, the previously-pending image index is
     returned via *displaced_idx so the caller can do the bookkeeping
     needed for a dropped frame (consume the dropped image's
     vk_render_done semaphore, signal its gl_sample_done so the next
     Acquire on it doesn't block). If the slot was empty, returns
     UINT32_MAX in *displaced_idx.

   present_id and desired_ns come from VkPresentTimesInfoGOOGLE on the
   app's VkPresentInfoKHR pNext; both zero when the app isn't using
   display-timing. The worker uses these to populate
   VK_GOOGLE_display_timing history. */
static void worker_post(Swapchain *sc, uint32_t idx,
                        uint32_t present_id, uint64_t desired_ns,
                        uint32_t *displaced_idx) {
    bool is_mailbox = (sc->present_mode == VK_PRESENT_MODE_MAILBOX_KHR);
    uint32_t displaced = UINT32_MAX;

    pthread_mutex_lock(&sc->worker_lock);

    if (!is_mailbox) {
        /* FIFO path: block until slot is free. */
        while (sc->worker_pending && sc->worker_running)
            pthread_cond_wait(&sc->worker_cv_done, &sc->worker_lock);
    } else if (sc->worker_pending) {
        /* MAILBOX path: a previous present hasn't been picked up yet.
           Replace it; the caller will handle the displaced image's
           semaphore cleanup. */
        displaced = sc->worker_pending_idx;
    }

    if (!sc->worker_running) {
        pthread_mutex_unlock(&sc->worker_lock);
        if (displaced_idx) *displaced_idx = UINT32_MAX;
        return;
    }
    sc->worker_pending = true;
    sc->worker_pending_idx = idx;
    sc->worker_pending_present_id = present_id;
    sc->worker_pending_desired_ns = desired_ns;
    pthread_cond_signal(&sc->worker_cv_pending);
    pthread_mutex_unlock(&sc->worker_lock);

    if (displaced_idx) *displaced_idx = displaced;
}

/* Tell the worker to exit and join the thread. */
static void worker_shutdown(Swapchain *sc) {
    pthread_mutex_lock(&sc->worker_lock);
    if (!sc->worker_running) { pthread_mutex_unlock(&sc->worker_lock); return; }
    sc->worker_quit = true;
    pthread_cond_broadcast(&sc->worker_cv_pending);
    pthread_mutex_unlock(&sc->worker_lock);
    pthread_join(sc->worker, NULL);
}

/* Resolve GL extension entrypoints. Must be called with the GLX context current. */
static bool resolve_gl_funcs(Swapchain *sc) {
#define GLX(t, name) sc->name = (t)glXGetProcAddressARB((const GLubyte*)#name)
    GLX(PFNGLXSWAPINTERVALEXTPROC,        glXSwapIntervalEXT);
    GLX(PFNGLCREATEMEMORYOBJECTSEXTPROC,  glCreateMemoryObjectsEXT);
    GLX(PFNGLIMPORTMEMORYFDEXTPROC,       glImportMemoryFdEXT);
    GLX(PFNGLTEXSTORAGEMEM2DEXTPROC,      glTexStorageMem2DEXT);
    GLX(PFNGLDELETEMEMORYOBJECTSEXTPROC,  glDeleteMemoryObjectsEXT);
    GLX(PFNGLGENSEMAPHORESEXTPROC,        glGenSemaphoresEXT);
    GLX(PFNGLIMPORTSEMAPHOREFDEXTPROC,    glImportSemaphoreFdEXT);
    GLX(PFNGLWAITSEMAPHOREEXTPROC,        glWaitSemaphoreEXT);
    GLX(PFNGLSIGNALSEMAPHOREEXTPROC,      glSignalSemaphoreEXT);
    GLX(PFNGLDELETESEMAPHORESEXTPROC,     glDeleteSemaphoresEXT);

    /* GLX_SGI_video_sync. Optional; we fall back to glXSwapBuffers's
       built-in vsync if not available. The functions take/return plain C
       int and unsigned int, so no GLX-headers typedefs needed; the casts
       to a local function-pointer type spell out the signatures. */
    sc->glXGetVideoSyncSGI  =
        (int (*)(unsigned *))         glXGetProcAddressARB((const GLubyte*)"glXGetVideoSyncSGI");
    sc->glXWaitVideoSyncSGI =
        (int (*)(int, int, unsigned*))glXGetProcAddressARB((const GLubyte*)"glXWaitVideoSyncSGI");
    if (sc->glXGetVideoSyncSGI && sc->glXWaitVideoSyncSGI) {
        LOG_INFO("GLX_SGI_video_sync present; will sleep in glXWaitVideoSyncSGI between swaps");
    } else {
        LOG_INFO("GLX_SGI_video_sync NOT present; the worker thread may spin during glXSwapBuffers's built-in vsync wait");
    }
#undef GLX
    if (!sc->glCreateMemoryObjectsEXT || !sc->glImportMemoryFdEXT || !sc->glTexStorageMem2DEXT
     || !sc->glGenSemaphoresEXT || !sc->glImportSemaphoreFdEXT
     || !sc->glWaitSemaphoreEXT || !sc->glSignalSemaphoreEXT) {
        LOG_ERR("missing GL_EXT_memory_object_fd or GL_EXT_semaphore_fd entrypoints");
        return false;
    }
    return true;
}

/* Compile the textured-quad blit program once per swapchain. */
static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(s, sizeof(buf), NULL, buf);
        LOG_ERR("shader compile: %s", buf);
        glDeleteShader(s); return 0;
    }
    return s;
}

static bool build_blit_program(Swapchain *sc) {
    const char *vs =
        "#version 330 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=1) in vec2 uv;\n"
        "out vec2 v_uv;\n"
        "void main(){ v_uv = uv; gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs =
        "#version 330 core\n"
        "in vec2 v_uv;\n"
        "out vec4 frag;\n"
        "uniform sampler2D tex;\n"
        "void main(){ frag = texture(tex, v_uv); }\n";
    GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
    GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
    if (!v || !f) return false;
    sc->prog = glCreateProgram();
    glAttachShader(sc->prog, v); glAttachShader(sc->prog, f);
    glLinkProgram(sc->prog);
    GLint ok = 0; glGetProgramiv(sc->prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(sc->prog, sizeof(buf), NULL, buf);
        LOG_ERR("program link: %s", buf);
        return false;
    }
    glDeleteShader(v); glDeleteShader(f);
    sc->u_tex = glGetUniformLocation(sc->prog, "tex");

    glGenVertexArrays(1, &sc->vao); glBindVertexArray(sc->vao);
    glGenBuffers(1, &sc->vbo); glBindBuffer(GL_ARRAY_BUFFER, sc->vbo);
    /* Quad in clip space, UVs flipped vertically because Vulkan's coordinate
       system and GL's differ in Y, and we want the on-screen image to match
       what the app rendered (top-of-image -> top-of-window). */
    float quad[] = {
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 0.0f,
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);
    return true;
}

/* ----------------------------------------------------------------------- */
/* Per-image setup / teardown                                              */
/* ----------------------------------------------------------------------- */

static int find_memtype(const VkPhysicalDeviceMemoryProperties *p, uint32_t bits,
                        VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < p->memoryTypeCount; i++)
        if ((bits & (1u << i)) && (p->memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

static VkResult create_exportable_image(DevNode *dev, Swapchain *sc, PerImage *pi) {
    DeviceDispatch *d = &dev->d;

    VkExternalMemoryImageCreateInfo emi = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.pNext = &emi;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = sc->format;
    ici.extent.width = sc->extent.width;
    ici.extent.height = sc->extent.height;
    ici.extent.depth = 1;
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    /* Honor whatever usage flags the app asked for on the swapchain, plus
       what we need to do our job: SAMPLED_BIT so GL can sample the image,
       COLOR_ATTACHMENT_BIT because the app will be rendering into it. The
       OR is intentional — if the app already asked for SAMPLED or other
       flags, we keep them. If they asked for INPUT_ATTACHMENT or STORAGE
       or whatever, we honor that too. Strip PRESENT_SRC-only flags that
       don't apply to our offscreen-allocated images. */
    ici.usage = sc->image_usage
              | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult r = d->CreateImage(dev->device, &ici, NULL, &pi->image);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements mreq;
    d->GetImageMemoryRequirements(dev->device, pi->image, &mreq);
    pi->size = mreq.size;

    int mt = find_memtype(&dev->memp, mreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt < 0) mt = find_memtype(&dev->memp, mreq.memoryTypeBits, 0);
    if (mt < 0) { d->DestroyImage(dev->device, pi->image, NULL); pi->image = VK_NULL_HANDLE; return VK_ERROR_OUT_OF_DEVICE_MEMORY; }

    VkExportMemoryAllocateInfo eai = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
    eai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryDedicatedAllocateInfo dai = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    dai.image = pi->image;
    eai.pNext = &dai;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.pNext = &eai; mai.allocationSize = mreq.size; mai.memoryTypeIndex = (uint32_t)mt;

    r = d->AllocateMemory(dev->device, &mai, NULL, &pi->memory);
    if (r != VK_SUCCESS) goto fail_img;
    r = d->BindImageMemory(dev->device, pi->image, pi->memory, 0);
    if (r != VK_SUCCESS) goto fail_mem;

    VkMemoryGetFdInfoKHR gfi = { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
    gfi.memory = pi->memory; gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    pi->mem_fd = -1;
    r = d->GetMemoryFdKHR(dev->device, &gfi, &pi->mem_fd);
    if (r != VK_SUCCESS || pi->mem_fd < 0) goto fail_mem;
    return VK_SUCCESS;

fail_mem:
    d->FreeMemory(dev->device, pi->memory, NULL); pi->memory = VK_NULL_HANDLE;
fail_img:
    d->DestroyImage(dev->device, pi->image, NULL); pi->image = VK_NULL_HANDLE;
    return r;
}

static VkResult create_exportable_semaphores(DevNode *dev, PerImage *pi) {
    DeviceDispatch *d = &dev->d;
    VkExportSemaphoreCreateInfo esci = { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    sci.pNext = &esci;
    VkResult r;
    r = d->CreateSemaphore(dev->device, &sci, NULL, &pi->vk_render_done);
    if (r != VK_SUCCESS) return r;
    r = d->CreateSemaphore(dev->device, &sci, NULL, &pi->gl_sample_done);
    if (r != VK_SUCCESS) goto fail_a;

    VkSemaphoreGetFdInfoKHR gfi = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
    gfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    gfi.semaphore = pi->vk_render_done;
    pi->vk_render_done_fd = -1;
    r = d->GetSemaphoreFdKHR(dev->device, &gfi, &pi->vk_render_done_fd);
    if (r != VK_SUCCESS || pi->vk_render_done_fd < 0) goto fail_b;
    gfi.semaphore = pi->gl_sample_done;
    pi->gl_sample_done_fd = -1;
    r = d->GetSemaphoreFdKHR(dev->device, &gfi, &pi->gl_sample_done_fd);
    if (r != VK_SUCCESS || pi->gl_sample_done_fd < 0) goto fail_b;

    /* Per-image acquire fence — used to block Acquire on CPU side until
       the worker has actually finished sampling this image (which is when
       gl_sample_done gets re-signaled by the worker's glSignalSemaphoreEXT).
       Start signaled so the first NIMG acquires don't block. */
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    r = d->CreateFence(dev->device, &fci, NULL, &pi->acquire_fence);
    if (r != VK_SUCCESS) goto fail_b;

    return VK_SUCCESS;

fail_b:
    d->DestroySemaphore(dev->device, pi->gl_sample_done, NULL); pi->gl_sample_done = VK_NULL_HANDLE;
fail_a:
    d->DestroySemaphore(dev->device, pi->vk_render_done, NULL); pi->vk_render_done = VK_NULL_HANDLE;
    return r;
}

static bool gl_import_image(Swapchain *sc, PerImage *pi) {
    sc->glCreateMemoryObjectsEXT(1, &pi->gl_memobj);
    if (glGetError() != GL_NO_ERROR) return false;
    int dup_fd = dup(pi->mem_fd);
    if (dup_fd < 0) return false;
    sc->glImportMemoryFdEXT(pi->gl_memobj, pi->size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, dup_fd);
    if (glGetError() != GL_NO_ERROR) return false;
    /* GL has imported (and now owns) its dup of the fd. Close the original
       so we drop OUR reference. The VkDeviceMemory handle still keeps the
       underlying memory alive on the Vulkan side. Keeping the original fd
       open is harmless on the desktop driver but on Tegra L4T r32.x it
       creates two parallel kernel-object references for the same underlying
       NvRm resource — and the driver's cleanup logic doesn't always handle
       that correctly under heavy submit traffic, leading to NvRmSyncClose
       crashing on a double-close. */
    close(pi->mem_fd);
    pi->mem_fd = -1;

    glGenTextures(1, &pi->gl_texture);
    glBindTexture(GL_TEXTURE_2D, pi->gl_texture);
    GLenum internal = GL_RGBA8;
    if (sc->format == VK_FORMAT_R8G8B8A8_SRGB || sc->format == VK_FORMAT_B8G8R8A8_SRGB)
        internal = GL_SRGB8_ALPHA8;
    sc->glTexStorageMem2DEXT(GL_TEXTURE_2D, 1, internal,
                             sc->extent.width, sc->extent.height, pi->gl_memobj, 0);
    if (glGetError() != GL_NO_ERROR) return false;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    /* The bytes in our image memory are laid out in the Vulkan format's
       channel order: BGRA for B8G8R8A8_*, RGBA for R8G8B8A8_*. GL's sampling
       always pretends the texture is RGBA. For BGRA-format swapchains we
       therefore have to remap R<->B at sample time so red and blue come
       from the right bytes. */
    if (sc->format == VK_FORMAT_B8G8R8A8_UNORM || sc->format == VK_FORMAT_B8G8R8A8_SRGB) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
    }
    /* D3D games (via DXVK) often leave the backbuffer alpha channel at 0
       (D3D's X8 formats have undefined alpha). If GL samples that alpha=0 and
       the child window has a 32-bit RGBA visual, the compositor blends the
       window as fully transparent — showing the blank parent window behind it.
       Force alpha=1 so the child window is always opaque to the compositor. */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE);
    return true;
}

static bool gl_import_semaphores(Swapchain *sc, PerImage *pi) {
    sc->glGenSemaphoresEXT(1, &pi->gl_render_sem);
    sc->glGenSemaphoresEXT(1, &pi->gl_sample_sem);
    int fd1 = dup(pi->vk_render_done_fd);
    int fd2 = dup(pi->gl_sample_done_fd);
    if (fd1 < 0 || fd2 < 0) return false;
    sc->glImportSemaphoreFdEXT(pi->gl_render_sem, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd1);
    if (glGetError() != GL_NO_ERROR) return false;
    sc->glImportSemaphoreFdEXT(pi->gl_sample_sem, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd2);
    if (glGetError() != GL_NO_ERROR) return false;
    /* Same reasoning as gl_import_image: close the originals now that GL owns
       its duplicates. The VkSemaphore handles keep the Vulkan-side semaphores
       alive; the fds are just transport handles for cross-API import and we
       no longer need them. On Tegra L4T r32.x, keeping them open alongside
       GL's duplicates is what causes NvRmSyncClose to crash under load
       (observed with PPSSPP and other high-submit-rate apps). */
    close(pi->vk_render_done_fd);
    close(pi->gl_sample_done_fd);
    pi->vk_render_done_fd = -1;
    pi->gl_sample_done_fd = -1;
    return true;
}

static void destroy_perimage(DevNode *dev, Swapchain *sc, PerImage *pi) {
    DeviceDispatch *d = &dev->d;
    if (pi->gl_texture) glDeleteTextures(1, &pi->gl_texture);
    if (pi->gl_memobj && sc->glDeleteMemoryObjectsEXT)
        sc->glDeleteMemoryObjectsEXT(1, &pi->gl_memobj);
    if (pi->gl_render_sem && sc->glDeleteSemaphoresEXT)
        sc->glDeleteSemaphoresEXT(1, &pi->gl_render_sem);
    if (pi->gl_sample_sem && sc->glDeleteSemaphoresEXT)
        sc->glDeleteSemaphoresEXT(1, &pi->gl_sample_sem);

    if (pi->vk_render_done) d->DestroySemaphore(dev->device, pi->vk_render_done, NULL);
    if (pi->gl_sample_done) d->DestroySemaphore(dev->device, pi->gl_sample_done, NULL);
    if (pi->acquire_fence)  d->DestroyFence    (dev->device, pi->acquire_fence, NULL);
    if (pi->image)          d->DestroyImage    (dev->device, pi->image, NULL);
    if (pi->memory)         d->FreeMemory      (dev->device, pi->memory, NULL);

    /* The fds we hold here are normally -1 by this point: we close them
       immediately after duping into GL (see gl_import_image and
       gl_import_semaphores for the rationale). The guards below only fire
       if setup partially failed before the imports happened. */
    if (pi->mem_fd            >= 0) close(pi->mem_fd);
    if (pi->vk_render_done_fd >= 0) close(pi->vk_render_done_fd);
    if (pi->gl_sample_done_fd >= 0) close(pi->gl_sample_done_fd);

    memset(pi, 0, sizeof(*pi));
    pi->mem_fd = pi->vk_render_done_fd = pi->gl_sample_done_fd = -1;
}

/* ----------------------------------------------------------------------- */
/* Surface bypass-compositor hint                                          */
/* ----------------------------------------------------------------------- */

/* ----------------------------------------------------------------------- */
/* Surface hooks                                                           */
/* ----------------------------------------------------------------------- */

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateXlibSurfaceKHR(VkInstance instance,
                            const VkXlibSurfaceCreateInfoKHR *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator,
                            VkSurfaceKHR *pSurface) {
    {
        InstNode *in = inst_lookup(dispatch_key(instance));
        if (g_layer_disabled || (in && in->passthrough))
            return in->d.CreateXlibSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }
    /* Surface query functions (GetPhysicalDeviceSurfaceCapabilitiesKHR etc.)
       call XGetGeometry and other X11 functions via g_libs.* pointers that
       are only populated by lib_load().  Those queries can arrive before
       CreateSwapchainKHR, so we must load the libraries here. */
    if (!lib_load()) {
        InstNode *in = inst_lookup(dispatch_key(instance));
        return in->d.CreateXlibSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }
    Surface *s = calloc(1, sizeof(*s));
    if (!s) return VK_ERROR_OUT_OF_HOST_MEMORY;
    s->magic = SURFACE_MAGIC;
    s->kind = SURF_XLIB;
    s->dpy = pCreateInfo->dpy;
    s->window = pCreateInfo->window;
    s->owns_dpy = false;
    /* Create a real ICD VkSurfaceKHR for this window alongside our wrapper.
       Non-NVIDIA devices (e.g. llvmpipe) don't know about our Surface*
       pointer; CreateSwapchainKHR uses this handle when forwarding to them. */
    {
        InstNode *in = inst_lookup(dispatch_key(instance));
        if (in && in->d.CreateXlibSurfaceKHR &&
            in->d.CreateXlibSurfaceKHR(instance, pCreateInfo, NULL,
                                        &s->icd_surface) == VK_SUCCESS)
            s->icd_inst = instance;
    }
    *pSurface = (VkSurfaceKHR)(uintptr_t)s;
    LOG_INFO("CreateXlibSurfaceKHR -> surface=%p dpy=%p win=0x%lx", s, s->dpy, s->window);
    return VK_SUCCESS;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateXcbSurfaceKHR(VkInstance instance,
                           const VkXcbSurfaceCreateInfoKHR *pCreateInfo,
                           const VkAllocationCallbacks *pAllocator,
                           VkSurfaceKHR *pSurface) {
    {
        InstNode *in = inst_lookup(dispatch_key(instance));
        if (g_layer_disabled || (in && in->passthrough))
            return in->d.CreateXcbSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }
    /* XCB-only apps don't give us an Xlib Display*. We need one for GLX
       (GLX is Xlib-bound). Open our own Display* over the same X server.
       lib_load() is deferred to CreateSwapchain; XOpenDisplay is resolved
       there too, so for the XCB path we load it eagerly here. */
    if (!lib_load()) {
        InstNode *in = inst_lookup(dispatch_key(instance));
        return in->d.CreateXcbSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }
    Surface *s = calloc(1, sizeof(*s));
    if (!s) return VK_ERROR_OUT_OF_HOST_MEMORY;
    s->magic = SURFACE_MAGIC;
    s->kind = SURF_XCB;
    s->dpy = XOpenDisplay(NULL);
    if (!s->dpy) { free(s); return VK_ERROR_INITIALIZATION_FAILED; }
    s->owns_dpy = true;
    s->window = pCreateInfo->window;
    /* Create a real ICD VkSurfaceKHR alongside our wrapper for non-NVIDIA
       device passthrough (see comment in CreateXlibSurfaceKHR). */
    {
        InstNode *in = inst_lookup(dispatch_key(instance));
        if (in && in->d.CreateXcbSurfaceKHR &&
            in->d.CreateXcbSurfaceKHR(instance, pCreateInfo, NULL,
                                       &s->icd_surface) == VK_SUCCESS)
            s->icd_inst = instance;
    }
    *pSurface = (VkSurfaceKHR)(uintptr_t)s;
    LOG_INFO("CreateXcbSurfaceKHR -> surface=%p dpy=%p (opened) win=0x%x",
             s, s->dpy, pCreateInfo->window);
    return VK_SUCCESS;
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL
layer_DestroySurfaceKHR(VkInstance instance, VkSurfaceKHR surface,
                         const VkAllocationCallbacks *pAllocator) {
    if (!surface) return;
    Surface *s = as_surface(surface);
    InstNode *in = inst_lookup(dispatch_key(instance));
    if (!s) {  /* not ours — was created as passthrough */
        in->d.DestroySurfaceKHR(instance, surface, pAllocator);
        return;
    }
    if (s->icd_surface && in)
        in->d.DestroySurfaceKHR(s->icd_inst, s->icd_surface, NULL);
    if (s->owns_dpy && s->dpy) XCloseDisplay(s->dpy);
    free(s);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice physicalDevice,
                                          uint32_t queueFamilyIndex,
                                          VkSurfaceKHR surface,
                                          VkBool32 *pSupported) {
    Surface *s = as_surface(surface);
    if (!s) {
        InstNode *in = inst_lookup(dispatch_key(physicalDevice));
        return in->d.GetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, pSupported);
    }
    /* Any graphics queue family supports our surface; we don't depend on
       Vulkan WSI presentation queues. */
    *pSupported = VK_TRUE;
    return VK_SUCCESS;
}

static void window_size(Surface *s, uint32_t *w, uint32_t *h) {
    Window root; int x, y; unsigned int ww, hh, b, dep;
    if (XGetGeometry(s->dpy, s->window, &root, &x, &y, &ww, &hh, &b, &dep)) {
        *w = ww; *h = hh;
    } else {
        *w = *h = 0;
    }
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice physicalDevice,
                                               VkSurfaceKHR surface,
                                               VkSurfaceCapabilitiesKHR *pCaps) {
    Surface *s = as_surface(surface);
    if (!s) {
        InstNode *in = inst_lookup(dispatch_key(physicalDevice));
        return in->d.GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pCaps);
    }
    uint32_t w = 0, h = 0; window_size(s, &w, &h);
    pCaps->minImageCount = MIN_IMAGES;
    pCaps->maxImageCount = MAX_IMAGES;
    pCaps->currentExtent.width  = w ? w : 1;
    pCaps->currentExtent.height = h ? h : 1;
    pCaps->minImageExtent.width  = 1;
    pCaps->minImageExtent.height = 1;
    pCaps->maxImageExtent.width  = 16384;
    pCaps->maxImageExtent.height = 16384;
    pCaps->maxImageArrayLayers = 1;
    pCaps->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    pCaps->currentTransform    = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    pCaps->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    /* Advertise everything common we can actually back. Apps will only use
       the bits they actually need, and we OR-include them in our image
       create. */
    pCaps->supportedUsageFlags =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
      | VK_IMAGE_USAGE_SAMPLED_BIT
      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
      | VK_IMAGE_USAGE_TRANSFER_DST_BIT
      | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
      | VK_IMAGE_USAGE_STORAGE_BIT;
    return VK_SUCCESS;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice physicalDevice,
                                          VkSurfaceKHR surface,
                                          uint32_t *pCount,
                                          VkSurfaceFormatKHR *pFormats) {
    Surface *s = as_surface(surface);
    if (!s) {
        InstNode *in = inst_lookup(dispatch_key(physicalDevice));
        return in->d.GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pCount, pFormats);
    }
    /* We support BGRA8 and RGBA8 UNORM and SRGB variants in OPTIMAL tiling
       with OPAQUE_FD export. These are the four formats we expose. */
    static const VkSurfaceFormatKHR formats[] = {
        { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_B8G8R8A8_SRGB,  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_R8G8B8A8_SRGB,  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
    };
    uint32_t avail = sizeof(formats) / sizeof(formats[0]);
    if (!pFormats) { *pCount = avail; return VK_SUCCESS; }
    uint32_t n = *pCount < avail ? *pCount : avail;
    memcpy(pFormats, formats, n * sizeof(VkSurfaceFormatKHR));
    *pCount = n;
    return n == avail ? VK_SUCCESS : VK_INCOMPLETE;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice physicalDevice,
                                               VkSurfaceKHR surface,
                                               uint32_t *pCount,
                                               VkPresentModeKHR *pModes) {
    Surface *s = as_surface(surface);
    if (!s) {
        InstNode *in = inst_lookup(dispatch_key(physicalDevice));
        return in->d.GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pCount, pModes);
    }
    static const VkPresentModeKHR modes[] = {
        VK_PRESENT_MODE_FIFO_KHR,
        VK_PRESENT_MODE_FIFO_RELAXED_KHR,
        VK_PRESENT_MODE_IMMEDIATE_KHR,
    };
    uint32_t avail = sizeof(modes) / sizeof(modes[0]);
    if (!pModes) { *pCount = avail; return VK_SUCCESS; }
    uint32_t n = *pCount < avail ? *pCount : avail;
    memcpy(pModes, modes, n * sizeof(VkPresentModeKHR));
    *pCount = n;
    return n == avail ? VK_SUCCESS : VK_INCOMPLETE;
}

/* VK_KHR_get_surface_capabilities2 — the "2" variants take a chain-extensible
   input struct (VkPhysicalDeviceSurfaceInfo2KHR) and write into a chain-extensible
   output. For our purposes we just delegate to the v1 implementation for our
   managed surfaces; we ignore any unknown pNext extensions on either side.
   For non-managed surfaces we pass through to the underlying driver. */
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPhysicalDeviceSurfaceCapabilities2KHR(VkPhysicalDevice physicalDevice,
                                                const VkPhysicalDeviceSurfaceInfo2KHR *pSurfaceInfo,
                                                VkSurfaceCapabilities2KHR *pCaps) {
    if (!pSurfaceInfo || !pCaps) return VK_ERROR_VALIDATION_FAILED_EXT;
    Surface *s = as_surface(pSurfaceInfo->surface);
    InstNode *in = inst_lookup(dispatch_key(physicalDevice));
    if (!s) {
        if (in && in->d.GetPhysicalDeviceSurfaceCapabilities2KHR)
            return in->d.GetPhysicalDeviceSurfaceCapabilities2KHR(physicalDevice, pSurfaceInfo, pCaps);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    /* Delegate to v1 for the core surfaceCapabilities. */
    return layer_GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice,
                                                         pSurfaceInfo->surface,
                                                         &pCaps->surfaceCapabilities);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPhysicalDeviceSurfaceFormats2KHR(VkPhysicalDevice physicalDevice,
                                           const VkPhysicalDeviceSurfaceInfo2KHR *pSurfaceInfo,
                                           uint32_t *pCount,
                                           VkSurfaceFormat2KHR *pFormats) {
    if (!pSurfaceInfo) return VK_ERROR_VALIDATION_FAILED_EXT;
    Surface *s = as_surface(pSurfaceInfo->surface);
    InstNode *in = inst_lookup(dispatch_key(physicalDevice));
    if (!s) {
        if (in && in->d.GetPhysicalDeviceSurfaceFormats2KHR)
            return in->d.GetPhysicalDeviceSurfaceFormats2KHR(physicalDevice, pSurfaceInfo, pCount, pFormats);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    /* Get the v1 formats, then wrap each into a VkSurfaceFormat2KHR. */
    static const VkSurfaceFormatKHR formats[] = {
        { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_B8G8R8A8_SRGB,  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_R8G8B8A8_SRGB,  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
    };
    uint32_t avail = sizeof(formats) / sizeof(formats[0]);
    if (!pFormats) { *pCount = avail; return VK_SUCCESS; }
    uint32_t n = *pCount < avail ? *pCount : avail;
    for (uint32_t i = 0; i < n; i++) {
        pFormats[i].sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR;
        pFormats[i].pNext = NULL;
        pFormats[i].surfaceFormat = formats[i];
    }
    *pCount = n;
    return n == avail ? VK_SUCCESS : VK_INCOMPLETE;
}

/* ----------------------------------------------------------------------- */
/* Swapchain creation                                                      */
/* ----------------------------------------------------------------------- */

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateSwapchainKHR(VkDevice device,
                          const VkSwapchainCreateInfoKHR *ci,
                          const VkAllocationCallbacks *pAlloc,
                          VkSwapchainKHR *pOut) {
    DevNode *dev = dev_lookup(dispatch_key(device));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;

    Surface *surf = as_surface(ci->surface);

    /* Non-NVIDIA passthrough device: forward to the ICD using the real ICD
       surface handle stored in our Surface* wrapper.  The ICD does not know
       about our Surface* pointer and would fault or return an error if given
       it directly. */
    if (dev->passthrough) {
        VkSwapchainCreateInfoKHR modci = *ci;
        if (surf && surf->icd_surface)
            modci.surface = surf->icd_surface;
        PFN_vkCreateSwapchainKHR icd_fn = (PFN_vkCreateSwapchainKHR)
            dev->d.GetDeviceProcAddr(device, "vkCreateSwapchainKHR");
        if (!icd_fn) return VK_ERROR_INITIALIZATION_FAILED;
        return icd_fn(device, &modci, pAlloc, pOut);
    }

    if (g_layer_disabled || !surf)
        return dev->d.CreateSwapchainKHR(device, ci, pAlloc, pOut);

    /* Lazy-load libX11, libGL, libGLX. We can't link these in at build time
       because doing so causes a recursive-mutex deadlock inside the Vulkan
       loader during vkCreateInstance — see the big block comment near the
       top of this file. CreateSwapchain is the first point we actually
       need them, and we're well past CreateInstance here, so no loader
       mutex is held. lib_load() is idempotent and thread-safe. */
    if (!lib_load()) {
        LOG_ERR("CreateSwapchainKHR: failed to load libGL/libX11; falling through");
        return dev->d.CreateSwapchainKHR(device, ci, pAlloc, pOut);
    }

    /* Clamp image count to our range. */
    uint32_t want = ci->minImageCount;
    if (want < MIN_IMAGES) want = MIN_IMAGES;
    if (want > MAX_IMAGES) want = MAX_IMAGES;

    /* Reject formats we don't support. */
    switch (ci->imageFormat) {
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_SRGB:
        break;
    default:
        LOG_WARN("CreateSwapchainKHR: unsupported format %d, falling through to passthrough", ci->imageFormat);
        return dev->d.CreateSwapchainKHR(device, ci, pAlloc, pOut);
    }

    Swapchain *sc = calloc(1, sizeof(*sc));
    if (!sc) return VK_ERROR_OUT_OF_HOST_MEMORY;
    sc->magic = SWAPCHAIN_MAGIC;
    sc->dev   = dev;
    sc->surf  = surf;
    sc->image_count = want;
    sc->extent  = ci->imageExtent;
    sc->format  = ci->imageFormat;
    sc->color_space = ci->imageColorSpace;
    sc->present_mode = ci->presentMode;
    sc->image_usage  = ci->imageUsage;
    pthread_mutex_init(&sc->lock, NULL);
    for (uint32_t i = 0; i < MAX_IMAGES; i++) {
        sc->images[i].mem_fd = sc->images[i].vk_render_done_fd = sc->images[i].gl_sample_done_fd = -1;
    }

    int screen = DefaultScreen(surf->dpy);

    /* Query the app's window depth so we can prefer a matching FBConfig.
       If the app's window is 32-bit RGBA (Chromium, GTK CSD apps, etc.),
       we want our child window to also be 32-bit RGBA so the compositor
       can blend its translucent regions properly. If the app's window is
       24-bit, a 24-bit FBConfig is fine. */
    int parent_depth = 0;
    {
        XWindowAttributes pwa = {0};
        if (XGetWindowAttributes(surf->dpy, surf->window, &pwa)) {
            parent_depth = pwa.depth;
        }
    }

    sc->fbcfg = pick_fbconfig(surf->dpy, screen, sc->format, parent_depth, &sc->visinfo);
    if (!sc->fbcfg || !sc->visinfo) {
        LOG_ERR("pick_fbconfig: no suitable FBConfig for format %d", sc->format);
        free(sc);
        return dev->d.CreateSwapchainKHR(device, ci, pAlloc, pOut);
    }
    LOG_INFO("FBConfig visual: depth=%d (parent depth=%d)", sc->visinfo->depth, parent_depth);

    /* Create the GLX child window inside the app's surface window. We use
       our own visual (from the FBConfig) so GLX is happy. The app's window
       still has whatever visual the app chose; we never touch it directly. */
    {
        uint32_t pw = 0, ph = 0; window_size(surf, &pw, &ph);
        if (pw == 0 || ph == 0) { pw = sc->extent.width; ph = sc->extent.height; }
        sc->win_w = (int)pw; sc->win_h = (int)ph;

        sc->child_colormap = XCreateColormap(surf->dpy, surf->window,
                                              sc->visinfo->visual, AllocNone);
        XSetWindowAttributes swa = {0};
        swa.colormap = sc->child_colormap;
        swa.background_pixel = 0;
        swa.border_pixel = 0;
        /* No event mask — we don't want to receive events on the child,
           and even if we did, they'd go to whoever owns the X event queue. */
        sc->child_window = XCreateWindow(surf->dpy, surf->window,
                                         0, 0, pw, ph, 0,
                                         sc->visinfo->depth, InputOutput,
                                         sc->visinfo->visual,
                                         CWColormap | CWBackPixel | CWBorderPixel,
                                         &swa);
        if (!sc->child_window) {
            LOG_ERR("XCreateWindow(child) failed");
            if (sc->child_colormap) XFreeColormap(surf->dpy, sc->child_colormap);
            if (sc->visinfo) XFree(sc->visinfo);
            free(sc);
            return dev->d.CreateSwapchainKHR(device, ci, pAlloc, pOut);
        }
        XMapWindow(surf->dpy, sc->child_window);
        XFlush(surf->dpy);
    }

    sc->glctx = create_glx_context(surf->dpy, sc->fbcfg);
    if (!sc->glctx) {
        LOG_ERR("create_glx_context failed");
        XDestroyWindow(surf->dpy, sc->child_window);
        XFreeColormap(surf->dpy, sc->child_colormap);
        if (sc->visinfo) XFree(sc->visinfo);
        free(sc);
        return dev->d.CreateSwapchainKHR(device, ci, pAlloc, pOut);
    }
    sc->glctx_owned = true;

    if (!glXMakeCurrent(surf->dpy, sc->child_window, sc->glctx)) {
        LOG_ERR("initial glXMakeCurrent failed");
        glXDestroyContext(surf->dpy, sc->glctx);
        XDestroyWindow(surf->dpy, sc->child_window);
        XFreeColormap(surf->dpy, sc->child_colormap);
        if (sc->visinfo) XFree(sc->visinfo);
        free(sc);
        return dev->d.CreateSwapchainKHR(device, ci, pAlloc, pOut);
    }

    if (!resolve_gl_funcs(sc)) goto fail_gl_setup;
    if (!build_blit_program(sc)) goto fail_gl_setup;

    /* Allocate per-image Vulkan resources and import them into GL. */
    for (uint32_t i = 0; i < sc->image_count; i++) {
        VkResult r = create_exportable_image(dev, sc, &sc->images[i]);
        if (r != VK_SUCCESS) { LOG_ERR("create_exportable_image[%u]: %d", i, r); goto fail_perimg; }
        r = create_exportable_semaphores(dev, &sc->images[i]);
        if (r != VK_SUCCESS) { LOG_ERR("create_exportable_semaphores[%u]: %d", i, r); goto fail_perimg; }
        if (!gl_import_image(sc, &sc->images[i]))       { LOG_ERR("gl_import_image[%u]",      i); goto fail_perimg; }
        if (!gl_import_semaphores(sc, &sc->images[i]))  { LOG_ERR("gl_import_semaphores[%u]", i); goto fail_perimg; }

        /* Pre-signal gl_sample_done so the first Acquire doesn't block. */
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &sc->images[i].gl_sample_done;
        VkResult psr = queue_submit_locked(dev, dev->graphics_queue, 1, &si, VK_NULL_HANDLE);
        if (psr != VK_SUCCESS) {
            LOG_ERR("pre-signal QueueSubmit for image %u failed: %d", i, psr);
            goto fail_perimg;
        }
    }

    /* Release the GLX context from this thread before the worker takes it. */
    glXMakeCurrent(surf->dpy, None, NULL);

    /* Start the worker. From this point on, the worker owns sc->glctx. */
    pthread_mutex_init(&sc->worker_lock, NULL);
    pthread_cond_init(&sc->worker_cv_pending, NULL);
    pthread_cond_init(&sc->worker_cv_done, NULL);
    pthread_mutex_init(&sc->timing_lock, NULL);
    /* Default refresh duration to 60Hz; the worker will refine this
       from observed SGI vblank intervals once it starts running. */
    sc->refresh_duration_ns = 16666667ULL;
    sc->worker_running = true;
    if (pthread_create(&sc->worker, NULL, worker_thread_main, sc) != 0) {
        LOG_ERR("pthread_create(worker) failed");
        sc->worker_running = false;
        goto fail_perimg;
    }

    track_swapchain(sc);

    *pOut = (VkSwapchainKHR)(uintptr_t)sc;
    LOG_INFO("CreateSwapchainKHR -> sc=%p %ux%u fmt=%d images=%u present_mode=%d (async worker)",
             sc, sc->extent.width, sc->extent.height, sc->format,
             sc->image_count, sc->present_mode);
    return VK_SUCCESS;

fail_perimg:
    for (uint32_t i = 0; i < sc->image_count; i++) destroy_perimage(dev, sc, &sc->images[i]);
fail_gl_setup:
    if (sc->vbo) glDeleteBuffers(1, &sc->vbo);
    if (sc->vao) glDeleteVertexArrays(1, &sc->vao);
    if (sc->prog) glDeleteProgram(sc->prog);
    glXMakeCurrent(surf->dpy, None, NULL);
    if (sc->glctx_owned && sc->glctx) glXDestroyContext(surf->dpy, sc->glctx);
    if (sc->child_window) XDestroyWindow(surf->dpy, sc->child_window);
    if (sc->child_colormap) XFreeColormap(surf->dpy, sc->child_colormap);
    if (sc->visinfo) XFree(sc->visinfo);
    free(sc);
    /* Soft fail: fall through to the real Vulkan WSI so the app still runs (with tearing). */
    return dev->d.CreateSwapchainKHR(device, ci, pAlloc, pOut);
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL
layer_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain,
                           const VkAllocationCallbacks *pAlloc) {
    if (!swapchain) return;
    Swapchain *sc = as_swapchain(swapchain);
    DevNode *dev = dev_lookup(dispatch_key(device));
    if (!sc) { if (dev) dev->d.DestroySwapchainKHR(device, swapchain, pAlloc); return; }

    /* Drain any in-flight work. */
    dev->d.DeviceWaitIdle(dev->device);

    untrack_swapchain(sc);

    /* Shut down the worker before touching GL resources — the worker owns the
       GLX context. After worker_shutdown returns, no thread has the context
       current; we can take it here for the final cleanup. */
    worker_shutdown(sc);

    pthread_mutex_lock(&sc->lock);
    if (sc->glctx) {
        glXMakeCurrent(sc->surf->dpy, sc->child_window, sc->glctx);
        for (uint32_t i = 0; i < sc->image_count; i++) destroy_perimage(dev, sc, &sc->images[i]);
        if (sc->vbo)  glDeleteBuffers(1, &sc->vbo);
        if (sc->vao)  glDeleteVertexArrays(1, &sc->vao);
        if (sc->prog) glDeleteProgram(sc->prog);
        glXMakeCurrent(sc->surf->dpy, None, NULL);
        if (sc->glctx_owned) glXDestroyContext(sc->surf->dpy, sc->glctx);
    }
    if (sc->child_window)   XDestroyWindow(sc->surf->dpy, sc->child_window);
    if (sc->child_colormap) XFreeColormap (sc->surf->dpy, sc->child_colormap);
    if (sc->visinfo) XFree(sc->visinfo);
    pthread_mutex_unlock(&sc->lock);
    pthread_mutex_destroy(&sc->lock);
    pthread_cond_destroy(&sc->worker_cv_pending);
    pthread_cond_destroy(&sc->worker_cv_done);
    pthread_mutex_destroy(&sc->worker_lock);
    pthread_mutex_destroy(&sc->timing_lock);
    memset(sc, 0, sizeof(*sc));
    free(sc);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain,
                             uint32_t *pCount, VkImage *pImages) {
    Swapchain *sc = as_swapchain(swapchain);
    if (!sc) {
        DevNode *dev = dev_lookup(dispatch_key(device));
        return dev->d.GetSwapchainImagesKHR(device, swapchain, pCount, pImages);
    }
    if (!pImages) { *pCount = sc->image_count; return VK_SUCCESS; }
    uint32_t n = *pCount < sc->image_count ? *pCount : sc->image_count;
    for (uint32_t i = 0; i < n; i++) pImages[i] = sc->images[i].image;
    *pCount = n;
    return n == sc->image_count ? VK_SUCCESS : VK_INCOMPLETE;
}

/* ----------------------------------------------------------------------- */
/* VK_GOOGLE_display_timing                                                */
/* ----------------------------------------------------------------------- */

/* Report the display's refresh cycle duration to the application.
   The value is set at swapchain creation; we use a sensible default of
   1/60Hz and the worker refines it from observed inter-vblank intervals
   as the app runs. */
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetRefreshCycleDurationGOOGLE(VkDevice device, VkSwapchainKHR swapchain,
                                     VkRefreshCycleDurationGOOGLE *pDisplayTimingProperties) {
    (void)device;
    Swapchain *sc = as_swapchain(swapchain);
    if (!sc || !pDisplayTimingProperties) return VK_ERROR_INITIALIZATION_FAILED;
    pDisplayTimingProperties->refreshDuration = sc->refresh_duration_ns;
    return VK_SUCCESS;
}

/* Return past presentation timing history. Standard count-query pattern:
   if pPresentationTimings is NULL, write the count of available entries
   to *pPresentationTimingCount. Otherwise, copy up to
   *pPresentationTimingCount entries into the output array and update
   the count to how many were actually written. Returned entries are
   removed from our ring. */
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPastPresentationTimingGOOGLE(VkDevice device, VkSwapchainKHR swapchain,
                                       uint32_t *pPresentationTimingCount,
                                       VkPastPresentationTimingGOOGLE *pPresentationTimings) {
    (void)device;
    Swapchain *sc = as_swapchain(swapchain);
    if (!sc || !pPresentationTimingCount) return VK_ERROR_INITIALIZATION_FAILED;

    pthread_mutex_lock(&sc->timing_lock);

    if (pPresentationTimings == NULL) {
        *pPresentationTimingCount = sc->timing_count;
        pthread_mutex_unlock(&sc->timing_lock);
        return VK_SUCCESS;
    }

    uint32_t want = *pPresentationTimingCount;
    uint32_t have = sc->timing_count;
    uint32_t out  = want < have ? want : have;
    for (uint32_t i = 0; i < out; i++) {
        uint32_t slot = (sc->timing_head + i) & 63;
        pPresentationTimings[i] = sc->timing_ring[slot];
    }
    /* Consume the entries we returned. */
    sc->timing_head  = (sc->timing_head + out) & 63;
    sc->timing_count = have - out;
    *pPresentationTimingCount = out;

    VkResult r = (out < have) ? VK_INCOMPLETE : VK_SUCCESS;
    pthread_mutex_unlock(&sc->timing_lock);
    return r;
}

/* ----------------------------------------------------------------------- */
/* Acquire / Present                                                       */
/* ----------------------------------------------------------------------- */

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_AcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain,
                           uint64_t timeout, VkSemaphore semaphore, VkFence fence,
                           uint32_t *pIndex) {
    Swapchain *sc = as_swapchain(swapchain);
    DevNode *dev = dev_lookup(dispatch_key(device));
    if (!sc) return dev->d.AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pIndex);

    pthread_mutex_lock(&sc->lock);
    /* Find the next free image. Round-robin among acquired==false images. */
    uint32_t idx = sc->next_acquire;
    uint32_t tried = 0;
    while (tried < sc->image_count && sc->images[idx].acquired) {
        idx = (idx + 1) % sc->image_count;
        tried++;
    }
    if (tried == sc->image_count) {
        pthread_mutex_unlock(&sc->lock);
        return VK_NOT_READY;
    }
    sc->images[idx].acquired = true;
    sc->next_acquire = (idx + 1) % sc->image_count;

    /* Reset our internal acquire_fence (it was either signaled-at-creation
       for the first N acquires, or signaled by the previous bridge submit). */
    dev->d.ResetFences(dev->device, 1, &sc->images[idx].acquire_fence);

    /* Bridge submit: wait on the per-image gl_sample_done semaphore, signal
       the app's requested acquire semaphore and/or fence, AND signal our
       internal acquire_fence. The fence lets us CPU-block here until the
       worker has actually finished sampling the image, which is the real
       backpressure point: without this, the app's loop runs unbounded
       (matching the worker's vsync rate) and burns CPU. */
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &sc->images[idx].gl_sample_done;
    si.pWaitDstStageMask    = &stage;
    if (semaphore) { si.signalSemaphoreCount = 1; si.pSignalSemaphores = &semaphore; }
    VkResult r = queue_submit_locked(dev, dev->graphics_queue, 1, &si, sc->images[idx].acquire_fence);
    pthread_mutex_unlock(&sc->lock);
    if (r != VK_SUCCESS) {
        LOG_ERR("AcquireNextImageKHR: bridge QueueSubmit failed: %d", r);
        return r;
    }

    /* CPU-block until our internal fence signals. This is the backpressure
       point that paces the app's render loop to actual presentation rate. */
    r = dev->d.WaitForFences(dev->device, 1, &sc->images[idx].acquire_fence, VK_TRUE, timeout);
    if (r == VK_TIMEOUT) return VK_TIMEOUT;
    if (r != VK_SUCCESS) {
        LOG_ERR("AcquireNextImageKHR: WaitForFences failed: %d", r);
        return r;
    }

    /* If the app passed its OWN fence (separate from our internal one), we
       need to also signal it. Easiest: do a second tiny submit that signals
       just the app's fence. Skip if no app fence. */
    if (fence) {
        VkSubmitInfo si2 = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        r = queue_submit_locked(dev, dev->graphics_queue, 1, &si2, fence);
        if (r != VK_SUCCESS) {
            LOG_ERR("AcquireNextImageKHR: app-fence signal submit failed: %d", r);
        }
    }

    *pIndex = idx;
    return VK_SUCCESS;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_QueueWaitIdle(VkQueue queue) {
    DevNode *dev = dev_lookup(dispatch_key(queue));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = dev->d.QueueWaitIdle(queue);
    if (r != VK_SUCCESS) {
        LOG_ERR("vkQueueWaitIdle FAILED: ret=%d queue=%p", r, (void*)queue);
    }
    return r;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_DeviceWaitIdle(VkDevice device) {
    DevNode *dev = dev_lookup(dispatch_key(device));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = dev->d.DeviceWaitIdle(device);
    if (r != VK_SUCCESS) {
        LOG_ERR("vkDeviceWaitIdle FAILED: ret=%d", r);
    }
    return r;
}

/* Track swapchains in a global list so we can answer "is this image ours" without
   knowing which swapchain it belongs to up front. */
#define MAX_TRACKED_SWAPCHAINS 16
static Swapchain *g_swapchains[MAX_TRACKED_SWAPCHAINS];
static pthread_mutex_t g_sc_lock = PTHREAD_MUTEX_INITIALIZER;

static void track_swapchain(Swapchain *sc) {
    pthread_mutex_lock(&g_sc_lock);
    for (int i = 0; i < MAX_TRACKED_SWAPCHAINS; i++) {
        if (!g_swapchains[i]) { g_swapchains[i] = sc; break; }
    }
    pthread_mutex_unlock(&g_sc_lock);
}
static void untrack_swapchain(Swapchain *sc) {
    pthread_mutex_lock(&g_sc_lock);
    for (int i = 0; i < MAX_TRACKED_SWAPCHAINS; i++) {
        if (g_swapchains[i] == sc) { g_swapchains[i] = NULL; break; }
    }
    pthread_mutex_unlock(&g_sc_lock);
}
/* Quick "is this image managed" check. Most barriers will hit non-managed
   images, so we want this to be as cheap as possible. The slowest part is
   acquiring the lock — but since the swapchain set rarely changes, we
   keep a generation counter and a thread-local cache to skip the lock
   entirely when nothing has changed. For simplicity, in the first pass we
   just take the lock and walk; if profiling shows this is still hot, the
   lock-free version is straightforward. */
static bool image_is_managed(VkImage img) {
    if (!img) return false;
    pthread_mutex_lock(&g_sc_lock);
    bool found = false;
    for (int i = 0; i < MAX_TRACKED_SWAPCHAINS && !found; i++) {
        Swapchain *sc = g_swapchains[i];
        if (!sc) continue;
        for (uint32_t j = 0; j < sc->image_count; j++) {
            if (sc->images[j].image == img) { found = true; break; }
        }
    }
    pthread_mutex_unlock(&g_sc_lock);
    return found;
}

/* Fast-path: are we tracking ANY swapchains at all? If not, no barriers can
   possibly hit our images, so skip the per-barrier check entirely. */
static bool any_swapchains_tracked(void) {
    pthread_mutex_lock(&g_sc_lock);
    bool any = false;
    for (int i = 0; i < MAX_TRACKED_SWAPCHAINS; i++) {
        if (g_swapchains[i]) { any = true; break; }
    }
    pthread_mutex_unlock(&g_sc_lock);
    return any;
}

/* Rewrite VK_IMAGE_LAYOUT_PRESENT_SRC_KHR (and SHARED_PRESENT_KHR) on barriers
   targeting our managed images. The driver can't transition a non-swapchain
   image to PRESENT_SRC — it tries to invoke present-engine metadata that doesn't
   exist for our externally-allocated images, and faults the GPU. We rewrite
   to SHADER_READ_ONLY_OPTIMAL, which is what we actually want anyway since GL
   will sample the image after the app is done with it. */
static VkImageLayout fix_layout(VkImageLayout l) {
    if (l == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (l == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR) return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return l;
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL
layer_CmdPipelineBarrier(VkCommandBuffer cb,
                          VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                          VkDependencyFlags depFlags,
                          uint32_t memBarrierCount, const VkMemoryBarrier *memBarriers,
                          uint32_t bufBarrierCount, const VkBufferMemoryBarrier *bufBarriers,
                          uint32_t imgBarrierCount, const VkImageMemoryBarrier *imgBarriers) {
    DevNode *dev = dev_lookup(dispatch_key(cb));
    if (!dev || !dev->d.CmdPipelineBarrier) return;

    /* Fast path: no image barriers, or no swapchains tracked. Pass through with
       no allocation, no lock, no copy. This is the path 99% of barriers take
       in a real application — the swapchain-image barriers are the rare case. */
    if (imgBarrierCount == 0 || !any_swapchains_tracked()) {
        dev->d.CmdPipelineBarrier(cb, srcStage, dstStage, depFlags,
                                   memBarrierCount, memBarriers,
                                   bufBarrierCount, bufBarriers,
                                   imgBarrierCount, imgBarriers);
        return;
    }

    /* Slow path: walk the image barriers checking for managed images. Only
       allocate a copy if at least one barrier actually needs rewriting. */
    bool any_managed = false;
    for (uint32_t i = 0; i < imgBarrierCount; i++) {
        if (image_is_managed(imgBarriers[i].image)) { any_managed = true; break; }
    }
    if (!any_managed) {
        dev->d.CmdPipelineBarrier(cb, srcStage, dstStage, depFlags,
                                   memBarrierCount, memBarriers,
                                   bufBarrierCount, bufBarriers,
                                   imgBarrierCount, imgBarriers);
        return;
    }

    VkImageMemoryBarrier *fixed = malloc(imgBarrierCount * sizeof(*fixed));
    if (!fixed) {
        /* Best effort: pass through unmodified. May fault but at least
           doesn't drop the barrier. */
        dev->d.CmdPipelineBarrier(cb, srcStage, dstStage, depFlags,
                                   memBarrierCount, memBarriers,
                                   bufBarrierCount, bufBarriers,
                                   imgBarrierCount, imgBarriers);
        return;
    }
    memcpy(fixed, imgBarriers, imgBarrierCount * sizeof(*fixed));
    for (uint32_t i = 0; i < imgBarrierCount; i++) {
        if (image_is_managed(fixed[i].image)) {
            fixed[i].oldLayout = fix_layout(fixed[i].oldLayout);
            fixed[i].newLayout = fix_layout(fixed[i].newLayout);
        }
    }
    dev->d.CmdPipelineBarrier(cb, srcStage, dstStage, depFlags,
                               memBarrierCount, memBarriers,
                               bufBarrierCount, bufBarriers,
                               imgBarrierCount, fixed);
    free(fixed);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_EndCommandBuffer(VkCommandBuffer cb) {
    DevNode *dev = dev_lookup(dispatch_key(cb));
    if (!dev || !dev->d.EndCommandBuffer) return VK_ERROR_INITIALIZATION_FAILED;
    return dev->d.EndCommandBuffer(cb);
}

/* vkCreateRenderPass intercept.

   When a render pass's attachment description has finalLayout (or
   initialLayout) set to PRESENT_SRC_KHR, the driver will perform an implicit
   layout transition at the end of the render pass execution. For our
   managed (non-WSI) swapchain images that's the same fault as a manual
   PRESENT_SRC barrier — the driver tries to invoke present-engine metadata
   that doesn't exist for externally-allocated images, and faults the GPU.

   Unlike with vkCmdPipelineBarrier, at vkCreateRenderPass time we don't
   know which images the render pass will be used with — that's determined
   later by the framebuffer. We pessimistically rewrite ANY PRESENT_SRC
   attachment layout to SHADER_READ_ONLY_OPTIMAL. This is safe because:
   - When our layer is active, we replace the swapchain entirely. There are
     no "real" presentable images in the application; all swapchain images
     are our managed ones, and they all want SHADER_READ_ONLY_OPTIMAL at
     present time anyway.
   - For non-swapchain attachments, applications shouldn't be specifying
     PRESENT_SRC anyway — that layout only exists for swapchain images.
     If an app does it incorrectly, our rewrite improves correctness. */
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateRenderPass(VkDevice device,
                        const VkRenderPassCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkRenderPass *pRenderPass) {
    DevNode *dev = dev_lookup(dispatch_key(device));
    if (!dev || !dev->d.CreateRenderPass) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pCreateInfo || pCreateInfo->attachmentCount == 0)
        return dev->d.CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);

    /* Scan for PRESENT_SRC attachments. */
    bool any = false;
    for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
        VkImageLayout il = pCreateInfo->pAttachments[i].initialLayout;
        VkImageLayout fl = pCreateInfo->pAttachments[i].finalLayout;
        if (il == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR || il == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR ||
            fl == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR || fl == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR) {
            any = true; break;
        }
    }
    if (!any) return dev->d.CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);

    VkAttachmentDescription *fixed = malloc(pCreateInfo->attachmentCount * sizeof(*fixed));
    if (!fixed) return dev->d.CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    memcpy(fixed, pCreateInfo->pAttachments,
           pCreateInfo->attachmentCount * sizeof(*fixed));
    for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
        fixed[i].initialLayout = fix_layout(fixed[i].initialLayout);
        fixed[i].finalLayout   = fix_layout(fixed[i].finalLayout);
    }
    VkRenderPassCreateInfo mod = *pCreateInfo;
    mod.pAttachments = fixed;
    VkResult r = dev->d.CreateRenderPass(device, &mod, pAllocator, pRenderPass);
    free(fixed);
    return r;
}

/* vkCreateRenderPass2 / vkCreateRenderPass2KHR intercept.
   Same PRESENT_SRC_KHR rewrite as v1, applied to the v2 struct.
   Vulkan 1.2 apps use this entrypoint instead of v1. */
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateRenderPass2(VkDevice device,
                         const VkRenderPassCreateInfo2 *pCreateInfo,
                         const VkAllocationCallbacks *pAllocator,
                         VkRenderPass *pRenderPass) {
    DevNode *dev = dev_lookup(dispatch_key(device));
    if (!dev || !dev->d.CreateRenderPass2) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pCreateInfo || pCreateInfo->attachmentCount == 0)
        return dev->d.CreateRenderPass2(device, pCreateInfo, pAllocator, pRenderPass);

    bool any = false;
    for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
        VkImageLayout il = pCreateInfo->pAttachments[i].initialLayout;
        VkImageLayout fl = pCreateInfo->pAttachments[i].finalLayout;
        if (il == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR || il == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR ||
            fl == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR || fl == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR) {
            any = true; break;
        }
    }
    if (!any) return dev->d.CreateRenderPass2(device, pCreateInfo, pAllocator, pRenderPass);

    VkAttachmentDescription2 *fixed = malloc(pCreateInfo->attachmentCount * sizeof(*fixed));
    if (!fixed) return dev->d.CreateRenderPass2(device, pCreateInfo, pAllocator, pRenderPass);
    memcpy(fixed, pCreateInfo->pAttachments,
           pCreateInfo->attachmentCount * sizeof(*fixed));
    for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
        fixed[i].initialLayout = fix_layout(fixed[i].initialLayout);
        fixed[i].finalLayout   = fix_layout(fixed[i].finalLayout);
    }
    VkRenderPassCreateInfo2 mod = *pCreateInfo;
    mod.pAttachments = fixed;
    VkResult r = dev->d.CreateRenderPass2(device, &mod, pAllocator, pRenderPass);
    free(fixed);
    return r;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateRenderPass2KHR(VkDevice device,
                             const VkRenderPassCreateInfo2 *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator,
                             VkRenderPass *pRenderPass) {
    /* KHR alias — same struct, same rewrite logic.  Both d.CreateRenderPass2
       and d.CreateRenderPass2KHR are set to whichever entrypoints the driver
       exposes (with fallback to each other), so layer_CreateRenderPass2 calls
       through correctly regardless of which name the driver uses. */
    return layer_CreateRenderPass2(device, pCreateInfo, pAllocator, pRenderPass);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_QueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence) {
    DevNode *dev = dev_lookup(dispatch_key(queue));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;

    static uint64_t submit_seq = 0;
    uint64_t my_seq = 0;
    if (g_diag_wait_after_submit) {
        my_seq = __atomic_add_fetch(&submit_seq, 1, __ATOMIC_SEQ_CST);
        LOG_INFO("DIAG QueueSubmit#%" PRIu64 ": queue=%p cnt=%u fence=%p [waits=%u cbs=%u sigs=%u]",
                 my_seq, (void*)queue, submitCount, (void*)fence,
                 pSubmits[0].waitSemaphoreCount,
                 pSubmits[0].commandBufferCount,
                 pSubmits[0].signalSemaphoreCount);
    }

    VkResult r = queue_submit_locked(dev, queue, submitCount, pSubmits, fence);
    if (r != VK_SUCCESS) {
        LOG_ERR("vkQueueSubmit FAILED at seq#%" PRIu64 ": ret=%d queue=%p submitCount=%u fence=%p",
                my_seq, r, (void*)queue, submitCount, (void*)fence);
        for (uint32_t i = 0; i < submitCount; i++) {
            LOG_ERR("  pSubmits[%u]: waitSem=%u cmdBuf=%u sigSem=%u",
                    i, pSubmits[i].waitSemaphoreCount,
                    pSubmits[i].commandBufferCount, pSubmits[i].signalSemaphoreCount);
        }
        return r;
    }

    if (g_diag_wait_after_submit) {
        VkResult wr = dev->d.DeviceWaitIdle(dev->device);
        if (wr != VK_SUCCESS) {
            LOG_ERR("DIAG DeviceWaitIdle AFTER QueueSubmit#%" PRIu64 " returned %d "
                    "(submit faulted: queue=%p cnt=%u cb_in_submit=%u)",
                    my_seq, wr, (void*)queue, submitCount,
                    submitCount > 0 ? pSubmits[0].commandBufferCount : 0);
        }
    }

    return r;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_WaitForFences(VkDevice device, uint32_t fenceCount, const VkFence *pFences,
                     VkBool32 waitAll, uint64_t timeout) {
    DevNode *dev = dev_lookup(dispatch_key(device));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = dev->d.WaitForFences(device, fenceCount, pFences, waitAll, timeout);
    if (r != VK_SUCCESS) {
        LOG_ERR("vkWaitForFences FAILED: ret=%d fenceCount=%u timeout=%" PRIu64,
                r, fenceCount, timeout);
    }
    return r;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_ResetFences(VkDevice device, uint32_t fenceCount, const VkFence *pFences) {
    DevNode *dev = dev_lookup(dispatch_key(device));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = dev->d.ResetFences(device, fenceCount, pFences);
    if (r != VK_SUCCESS) LOG_ERR("vkResetFences FAILED: ret=%d", r);
    return r;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pInfo) {
    /* Queue's dispatch key is the parent device's. */
    DevNode *dev = dev_lookup(dispatch_key(queue));
    if (!dev) return VK_ERROR_INITIALIZATION_FAILED;
    if (g_layer_disabled) {
        pthread_mutex_lock(&dev->submit_lock);
        VkResult r = dev->d.QueuePresentKHR(queue, pInfo);
        pthread_mutex_unlock(&dev->submit_lock);
        return r;
    }

    LOG_DBG("Present entry: queue=%p (graphics_queue=%p) swapchains=%u waitSems=%u",
             (void*)queue, (void*)dev->graphics_queue,
             pInfo->swapchainCount, pInfo->waitSemaphoreCount);

    /* Walk the pNext chain looking for VkPresentTimesInfoGOOGLE. If
       found, its pTimes[s] gives presentID and desiredPresentTime for
       swapchain index s. swapchainCount in that struct must equal the
       outer pInfo->swapchainCount per the extension spec; we trust the
       app on that. */
    const VkPresentTimesInfoGOOGLE *times_info = NULL;
    {
        const VkBaseInStructure *p = (const VkBaseInStructure *)pInfo->pNext;
        while (p) {
            if (p->sType == VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE) {
                times_info = (const VkPresentTimesInfoGOOGLE *)p;
                break;
            }
            p = p->pNext;
        }
    }

    VkResult overall = VK_SUCCESS;

    for (uint32_t s = 0; s < pInfo->swapchainCount; s++) {
        Swapchain *sc = as_swapchain(pInfo->pSwapchains[s]);
        if (!sc) {
            /* Mixed batch — submit just this one through real WSI. */
            VkPresentInfoKHR sub = *pInfo;
            sub.swapchainCount = 1;
            sub.pSwapchains    = &pInfo->pSwapchains[s];
            sub.pImageIndices  = &pInfo->pImageIndices[s];
            sub.pResults       = pInfo->pResults ? &pInfo->pResults[s] : NULL;
            sub.waitSemaphoreCount = (s == 0) ? pInfo->waitSemaphoreCount : 0;
            sub.pWaitSemaphores    = (s == 0) ? pInfo->pWaitSemaphores    : NULL;
            pthread_mutex_lock(&dev->submit_lock);
            VkResult r = dev->d.QueuePresentKHR(queue, &sub);
            pthread_mutex_unlock(&dev->submit_lock);
            if (pInfo->pResults) pInfo->pResults[s] = r;
            if (r != VK_SUCCESS && overall == VK_SUCCESS) overall = r;
            continue;
        }
        uint32_t idx = pInfo->pImageIndices[s];
        if (idx >= sc->image_count) {
            VkResult r = VK_ERROR_OUT_OF_DATE_KHR;
            if (pInfo->pResults) pInfo->pResults[s] = r;
            if (overall == VK_SUCCESS) overall = r;
            continue;
        }

        pthread_mutex_lock(&sc->lock);

        /* Bridge: wait on the app's render-done semaphore(s), signal our
           per-image vk_render_done. Only do this for the first swapchain in
           the batch; subsequent ones don't get the app's waitSemaphores
           applied per WSI semantics. */
        VkPipelineStageFlags stages[16];
        for (int i = 0; i < 16; i++) stages[i] = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        VkSubmitInfo bridge = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        if (s == 0 && pInfo->waitSemaphoreCount > 0) {
            bridge.waitSemaphoreCount = pInfo->waitSemaphoreCount;
            bridge.pWaitSemaphores    = pInfo->pWaitSemaphores;
            bridge.pWaitDstStageMask  = stages;
        }
        bridge.signalSemaphoreCount = 1;
        bridge.pSignalSemaphores = &sc->images[idx].vk_render_done;
        VkResult rb = queue_submit_locked(dev, queue, 1, &bridge, VK_NULL_HANDLE);
        if (rb != VK_SUCCESS) {
            pthread_mutex_unlock(&sc->lock);
            LOG_ERR("QueuePresentKHR: bridge QueueSubmit failed: %d (queue=%p idx=%u waitSems=%u signal=%p)",
                    rb, (void*)queue, idx, pInfo->waitSemaphoreCount,
                    (void*)sc->images[idx].vk_render_done);
            if (pInfo->pResults) pInfo->pResults[s] = rb;
            if (overall == VK_SUCCESS) overall = rb;
            continue;
        }

        /* The GL side runs in the worker thread which owns the GLX context.
           In FIFO mode this call blocks if the worker is still busy with
           the previous image — that's the natural backpressure point.
           In MAILBOX mode this call returns immediately and reports back
           the index of any image whose place we just took, so we can
           clean up its dangling semaphores before the next iteration. */
        sc->images[idx].acquired = false;
        pthread_mutex_unlock(&sc->lock);

        uint32_t present_id = 0;
        uint64_t desired_ns = 0;
        if (times_info && times_info->pTimes && s < times_info->swapchainCount) {
            present_id = times_info->pTimes[s].presentID;
            desired_ns = times_info->pTimes[s].desiredPresentTime;
        }
        uint32_t displaced_idx = UINT32_MAX;
        worker_post(sc, idx, present_id, desired_ns, &displaced_idx);

        /* MAILBOX drop bookkeeping. If worker_post returned a displaced
           index, that image was Presented earlier but its turn at the
           GLX swap never came. Two things need to happen for its state
           to be consistent for future use:

             1. Consume its vk_render_done semaphore. We had already
                signalled it via the bridge submit for that earlier
                Present. The worker would normally consume it via
                glWaitSemaphoreEXT during its loop iteration; since
                that iteration is skipped we have to consume here, or
                a future Present's bridge submit attempting to signal
                the same binary semaphore is undefined behaviour.

             2. Signal its gl_sample_done semaphore. The next time the
                app calls Acquire on this image, the acquire bridge
                waits on gl_sample_done. Without us signalling it the
                app would deadlock waiting for a frame that never
                actually rendered.

           One bridge submit handles both. The wait/signal happen on
           the queue serially with no actual GPU work between them; the
           submit returns fast. */
        if (displaced_idx != UINT32_MAX) {
            VkPipelineStageFlags drop_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            VkSubmitInfo drop_si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
            drop_si.waitSemaphoreCount   = 1;
            drop_si.pWaitSemaphores      = &sc->images[displaced_idx].vk_render_done;
            drop_si.pWaitDstStageMask    = &drop_stage;
            drop_si.signalSemaphoreCount = 1;
            drop_si.pSignalSemaphores    = &sc->images[displaced_idx].gl_sample_done;
            VkResult rd = queue_submit_locked(dev, queue, 1, &drop_si, VK_NULL_HANDLE);
            if (rd != VK_SUCCESS) {
                LOG_ERR("QueuePresentKHR: mailbox drop cleanup submit failed: %d "
                        "(displaced_idx=%u)", rd, displaced_idx);
                /* Don't propagate to the app — the present that displaced
                   this one is still in flight and is what the app actually
                   wanted. Best effort. */
            }
        }

        if (pInfo->pResults) pInfo->pResults[s] = VK_SUCCESS;
    }
    return overall;
}

/* ----------------------------------------------------------------------- */
/* Device / instance lifecycle                                             */
/* ----------------------------------------------------------------------- */

/* Forward decls. */
static PFN_vkVoidFunction layer_intercept_instance(const char *name);
static PFN_vkVoidFunction layer_intercept_device  (const char *name);
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
layer_GetDeviceProcAddr(VkDevice dev, const char *name);
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
layer_GetInstanceProcAddr(VkInstance inst, const char *name);

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateInstance(const VkInstanceCreateInfo *ci,
                      const VkAllocationCallbacks *pAlloc,
                      VkInstance *pInst) {
    VkLayerInstanceCreateInfo *lci = (VkLayerInstanceCreateInfo *)ci->pNext;
    while (lci && !(lci->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                    lci->function == VK_LAYER_LINK_INFO))
        lci = (VkLayerInstanceCreateInfo *)lci->pNext;
    if (!lci) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = lci->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    lci->u.pLayerInfo = lci->u.pLayerInfo->pNext;

    /* Like with CreateDevice, the app may not request the instance-level
       external memory/semaphore capability extensions. They're core in
       Vulkan 1.1 but still need to be listed if the app requested 1.0.
       Add them if the app didn't, harmlessly redundant if it did. */
    static const char *required_inst[] = {
        "VK_KHR_external_memory_capabilities",
        "VK_KHR_external_semaphore_capabilities",
        "VK_KHR_get_physical_device_properties2",
    };
    static const uint32_t n_required_inst = 3;
    uint32_t to_add = 0;
    bool need[3] = { true, true, true };
    for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) {
        for (uint32_t j = 0; j < n_required_inst; j++) {
            if (need[j] && !strcmp(ci->ppEnabledExtensionNames[i], required_inst[j])) need[j] = false;
        }
    }
    for (uint32_t j = 0; j < n_required_inst; j++) if (need[j]) to_add++;

    VkInstanceCreateInfo modci = *ci;
    const char **new_exts = NULL;
    if (to_add > 0) {
        uint32_t total = ci->enabledExtensionCount + to_add;
        new_exts = malloc(total * sizeof(const char *));
        if (!new_exts) return VK_ERROR_OUT_OF_HOST_MEMORY;
        memcpy(new_exts, ci->ppEnabledExtensionNames,
               ci->enabledExtensionCount * sizeof(const char *));
        uint32_t k = ci->enabledExtensionCount;
        for (uint32_t j = 0; j < n_required_inst; j++) if (need[j]) new_exts[k++] = required_inst[j];
        modci.enabledExtensionCount = total;
        modci.ppEnabledExtensionNames = new_exts;
    }

    PFN_vkCreateInstance next_create = (PFN_vkCreateInstance)next_gipa(NULL, "vkCreateInstance");
    VkResult r = next_create(&modci, pAlloc, pInst);
    if (new_exts) free(new_exts);
    if (r != VK_SUCCESS) return r;

    InstNode *node = calloc(1, sizeof(*node));
    if (!node) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    node->instance = *pInst;
    node->key = dispatch_key(*pInst);
#define I(name) node->d.name = (PFN_vk##name)next_gipa(*pInst, "vk" #name)
    I(GetInstanceProcAddr);
    I(DestroyInstance);
    I(EnumeratePhysicalDevices);
    I(GetPhysicalDeviceProperties);
    I(GetPhysicalDeviceMemoryProperties);
    I(GetPhysicalDeviceQueueFamilyProperties);
    I(CreateXlibSurfaceKHR);
    I(CreateXcbSurfaceKHR);
    I(DestroySurfaceKHR);
    I(GetPhysicalDeviceSurfaceSupportKHR);
    I(GetPhysicalDeviceSurfaceCapabilitiesKHR);
    I(GetPhysicalDeviceSurfaceFormatsKHR);
    I(GetPhysicalDeviceSurfacePresentModesKHR);
    I(GetPhysicalDeviceSurfaceCapabilities2KHR);
    I(GetPhysicalDeviceSurfaceFormats2KHR);
    I(GetPhysicalDeviceImageFormatProperties2);
    I(GetPhysicalDeviceExternalSemaphoreProperties);
#undef I
    /* Determine whether this instance has any NVIDIA physical device.
       We check now — before any surfaces are created — so that surface
       creation functions can immediately pass through for non-NVIDIA
       instances.  If EnumeratePhysicalDevices fails or returns zero
       devices we leave passthrough=false (default active) as a safe
       fallback; the missing-entrypoints guard in CreateDevice will catch
       any incompatibility later. */
    if (node->d.EnumeratePhysicalDevices && node->d.GetPhysicalDeviceProperties) {
        uint32_t nphys = 0;
        node->d.EnumeratePhysicalDevices(*pInst, &nphys, NULL);
        VkPhysicalDevice *physdevs = nphys ? calloc(nphys, sizeof(*physdevs)) : NULL;
        if (physdevs) {
            node->d.EnumeratePhysicalDevices(*pInst, &nphys, physdevs);
            bool has_nvidia = false;
            for (uint32_t i = 0; i < nphys && !has_nvidia; i++) {
                VkPhysicalDeviceProperties props = {0};
                node->d.GetPhysicalDeviceProperties(physdevs[i], &props);
                if (props.vendorID == 0x10DE) has_nvidia = true;
            }
            free(physdevs);
            if (!has_nvidia) {
                node->passthrough = true;
                LOG_INFO("CreateInstance: no NVIDIA device found — layer transparent for inst=%p", *pInst);
            }
        }
    }

    inst_insert(node);

    layer_check_disabled();
    LOG_INFO("CreateInstance: inst=%p%s", *pInst, g_layer_disabled ? " (disabled)" : "");
    return VK_SUCCESS;
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL
layer_DestroyInstance(VkInstance inst, const VkAllocationCallbacks *pAlloc) {
    InstNode *node = inst_lookup(dispatch_key(inst));
    if (!node) return;
    node->d.DestroyInstance(inst, pAlloc);
    inst_remove(dispatch_key(inst));
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo *ci,
                    const VkAllocationCallbacks *pAlloc, VkDevice *pDev) {
    VkLayerDeviceCreateInfo *lci = (VkLayerDeviceCreateInfo *)ci->pNext;
    while (lci && !(lci->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                    lci->function == VK_LAYER_LINK_INFO))
        lci = (VkLayerDeviceCreateInfo *)lci->pNext;
    if (!lci) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = lci->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   next_gdpa = lci->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    lci->u.pLayerInfo = lci->u.pLayerInfo->pNext;

    /* Vendor check: this layer is designed exclusively for NVIDIA Tegra
       (vendorID 0x10DE).  For any other driver — llvmpipe, Mesa radv,
       etc. — pass through entirely without modifying the extension list
       or tracking the device.  This prevents the layer from injecting
       extensions the driver doesn't support and from crashing on missing
       entrypoints. */
    {
        InstNode *inst_hint = NULL;
        pthread_mutex_lock(&g_inst_lock);
        for (int b = 0; b < HASH_BUCKETS && !inst_hint; b++)
            for (InstNode *n = g_inst_table[b]; n; n = n->next) { inst_hint = n; break; }
        pthread_mutex_unlock(&g_inst_lock);

        if (inst_hint && inst_hint->d.GetPhysicalDeviceProperties) {
            VkPhysicalDeviceProperties props = {0};
            inst_hint->d.GetPhysicalDeviceProperties(phys, &props);
            if (props.vendorID != 0x10DE) {
                LOG_INFO("CreateDevice: non-NVIDIA device (vendor=0x%04x, '%s') — layer transparent for this device",
                         props.vendorID, props.deviceName);
                /* Call through with the original (unmodified) extension list.
                   Then create a minimal passthrough DevNode that stores
                   next_gdpa.  This is essential: our GDPA always runs first
                   in the chain, and without a DevNode it would return NULL
                   for non-intercepted functions (e.g. vkGetDeviceQueue),
                   leaving the app with NULL function pointers. */
                PFN_vkCreateDevice next_cd =
                    (PFN_vkCreateDevice)next_gipa(VK_NULL_HANDLE, "vkCreateDevice");
                if (!next_cd) return VK_ERROR_INITIALIZATION_FAILED;
                VkResult r = next_cd(phys, ci, pAlloc, pDev);
                if (r != VK_SUCCESS) return r;

                DevNode *pt = calloc(1, sizeof(*pt));
                if (!pt) return VK_ERROR_OUT_OF_HOST_MEMORY;
                pt->device      = *pDev;
                pt->key         = dispatch_key(*pDev);
                pt->passthrough = true;
                /* Store next_gdpa in d.GetDeviceProcAddr; GDPA uses it to
                   forward all lookups to the next layer for this device. */
                pt->d.GetDeviceProcAddr = next_gdpa;
                dev_insert(pt);
                return VK_SUCCESS;
            }
        }
    }

    /* The application's vkCreateDevice doesn't enable the extensions we
       depend on (VK_KHR_external_memory_fd, VK_KHR_external_semaphore_fd,
       etc.). We need them for the present path. Build a modified
       VkDeviceCreateInfo with our extensions appended if not already
       present, then call through with that. */
    static const char *required[] = {
        "VK_KHR_external_memory",
        "VK_KHR_external_memory_fd",
        "VK_KHR_external_semaphore",
        "VK_KHR_external_semaphore_fd",
        "VK_KHR_dedicated_allocation",
        "VK_KHR_get_memory_requirements2",
    };
    static const uint32_t n_required = sizeof(required) / sizeof(required[0]);

    /* Count how many of our required extensions the app didn't already enable. */
    uint32_t to_add = 0;
    bool need[6] = { true, true, true, true, true, true };
    for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) {
        for (uint32_t j = 0; j < n_required; j++) {
            if (need[j] && !strcmp(ci->ppEnabledExtensionNames[i], required[j])) {
                need[j] = false;
            }
        }
    }
    for (uint32_t j = 0; j < n_required; j++) if (need[j]) to_add++;

    /* Extensions this LAYER provides to applications but which the
       underlying ICD does NOT implement. We must strip these from the
       extension list before passing CreateDevice down — otherwise the
       ICD rejects device creation with ERROR_EXTENSION_NOT_PRESENT.
       Apps that enable these extensions still get them: our intercepts
       are wired in at GetDeviceProcAddr time regardless of what the ICD
       says. */
    static const char *layer_provided[] = {
        "VK_GOOGLE_display_timing",
    };
    static const uint32_t n_layer_provided = 1;

    /* Build the down-going extension list: app's list minus any
       layer-provided names, plus our required extensions if not
       already present. We always allocate a new array since we may need
       to delete entries even if to_add is zero. */
    VkDeviceCreateInfo modci = *ci;
    uint32_t copy_n = 0;
    const char **new_exts = malloc(
        (ci->enabledExtensionCount + to_add) * sizeof(const char *));
    if (!new_exts) return VK_ERROR_OUT_OF_HOST_MEMORY;
    for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) {
        bool strip = false;
        for (uint32_t j = 0; j < n_layer_provided; j++) {
            if (!strcmp(ci->ppEnabledExtensionNames[i], layer_provided[j])) { strip = true; break; }
        }
        if (!strip) new_exts[copy_n++] = ci->ppEnabledExtensionNames[i];
    }
    for (uint32_t j = 0; j < n_required; j++) if (need[j]) new_exts[copy_n++] = required[j];
    modci.enabledExtensionCount = copy_n;
    modci.ppEnabledExtensionNames = new_exts;
    if (to_add > 0) {
        LOG_INFO("CreateDevice: appending %u required extensions", to_add);
    }

    PFN_vkCreateDevice next_create = (PFN_vkCreateDevice)next_gipa(VK_NULL_HANDLE, "vkCreateDevice");
    VkResult r = next_create(phys, &modci, pAlloc, pDev);
    if (new_exts) free(new_exts);
    if (r != VK_SUCCESS) {
        LOG_ERR("next CreateDevice failed: %d (extension support missing?)", r);
        return r;
    }

    InstNode *in = NULL;
    /* Find the instance whose physical device list contains 'phys'. We don't
       track that explicitly; just take the only instance in our table.
       This is brittle for multi-instance apps but matches our actual use. */
    pthread_mutex_lock(&g_inst_lock);
    for (int b = 0; b < HASH_BUCKETS && !in; b++)
        for (InstNode *n = g_inst_table[b]; n; n = n->next) { in = n; break; }
    pthread_mutex_unlock(&g_inst_lock);

    DevNode *node = calloc(1, sizeof(*node));
    if (!node) return VK_ERROR_OUT_OF_HOST_MEMORY;
    node->device = *pDev;
    node->phys = phys;
    node->key = dispatch_key(*pDev);
    node->idisp = in ? &in->d : NULL;
    node->inst = in ? in->instance : VK_NULL_HANDLE;

#define D(name) node->d.name = (PFN_vk##name)next_gdpa(*pDev, "vk" #name)
    D(GetDeviceProcAddr);
    D(DestroyDevice);
    D(DeviceWaitIdle);
    D(GetDeviceQueue);
    D(QueueSubmit);
    D(QueueWaitIdle);
    D(CreateImage);
    D(DestroyImage);
    D(GetImageMemoryRequirements);
    D(AllocateMemory);
    D(FreeMemory);
    D(BindImageMemory);
    D(CreateSemaphore);
    D(DestroySemaphore);
    D(CreateFence);
    D(DestroyFence);
    D(ResetFences);
    D(WaitForFences);
    D(GetFenceStatus);
    D(GetMemoryFdKHR);
    D(GetSemaphoreFdKHR);
    D(CmdPipelineBarrier);
    D(EndCommandBuffer);
    D(CreateRenderPass);
    /* Vulkan 1.2 render pass v2: core name first, KHR alias as fallback
       for drivers that only expose the extension entrypoint. */
    D(CreateRenderPass2);
    if (!node->d.CreateRenderPass2)
        node->d.CreateRenderPass2 = (PFN_vkCreateRenderPass2)next_gdpa(*pDev, "vkCreateRenderPass2KHR");
    D(CreateRenderPass2KHR);
    if (!node->d.CreateRenderPass2KHR)
        node->d.CreateRenderPass2KHR = node->d.CreateRenderPass2;
    D(CreateSwapchainKHR);
    D(DestroySwapchainKHR);
    D(GetSwapchainImagesKHR);
    D(AcquireNextImageKHR);
    D(QueuePresentKHR);
#undef D

    /* Sanity check: any of these being NULL means the next layer / ICD
       didn't expose them, which means our present path can't function.
       Don't crash — set the disabled flag so subsequent swapchain hooks
       fall through to passthrough. */
    if (!node->d.GetMemoryFdKHR || !node->d.GetSemaphoreFdKHR
     || !node->d.CreateSwapchainKHR) {
        LOG_WARN("CreateDevice: required entrypoints missing (GetMemoryFdKHR=%p GetSemaphoreFdKHR=%p CreateSwapchainKHR=%p); disabling layer for this device",
                 (void*)node->d.GetMemoryFdKHR,
                 (void*)node->d.GetSemaphoreFdKHR,
                 (void*)node->d.CreateSwapchainKHR);
        g_layer_disabled = true;
    }

    if (in) in->d.GetPhysicalDeviceMemoryProperties(phys, &node->memp);

    /* Find the first graphics-capable queue family and cache it. */
    uint32_t nqf = 0;
    if (in) in->d.GetPhysicalDeviceQueueFamilyProperties(phys, &nqf, NULL);
    VkQueueFamilyProperties *qfs = nqf ? calloc(nqf, sizeof(*qfs)) : NULL;
    if (in && qfs) in->d.GetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfs);
    node->graphics_qfi = 0;
    for (uint32_t i = 0; i < nqf; i++)
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { node->graphics_qfi = i; break; }
    free(qfs);
    node->d.GetDeviceQueue(*pDev, node->graphics_qfi, 0, &node->graphics_queue);
    pthread_mutex_init(&node->submit_lock, NULL);

    dev_insert(node);
    LOG_INFO("CreateDevice: dev=%p qfi=%u", *pDev, node->graphics_qfi);
    return VK_SUCCESS;
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL
layer_DestroyDevice(VkDevice dev, const VkAllocationCallbacks *pAlloc) {
    DevNode *node = dev_lookup(dispatch_key(dev));
    if (!node) return;
    if (node->passthrough) {
        /* Passthrough device: look up the real DestroyDevice via the stored
           next_gdpa, remove our DevNode, then call through. */
        PFN_vkDestroyDevice real_destroy =
            (PFN_vkDestroyDevice)node->d.GetDeviceProcAddr(dev, "vkDestroyDevice");
        dev_remove(dispatch_key(dev));
        if (real_destroy) real_destroy(dev, pAlloc);
        return;
    }
    node->d.DestroyDevice(dev, pAlloc);
    pthread_mutex_destroy(&node->submit_lock);
    dev_remove(dispatch_key(dev));
}

/* ----------------------------------------------------------------------- */
/* Entrypoint dispatch                                                     */
/* ----------------------------------------------------------------------- */

static PFN_vkVoidFunction layer_intercept_instance(const char *name) {
#define MATCH(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)layer_##n
    MATCH(GetInstanceProcAddr);
    MATCH(CreateInstance);
    MATCH(DestroyInstance);
    MATCH(CreateDevice);
    MATCH(CreateXlibSurfaceKHR);
    MATCH(CreateXcbSurfaceKHR);
    MATCH(DestroySurfaceKHR);
    MATCH(GetPhysicalDeviceSurfaceSupportKHR);
    MATCH(GetPhysicalDeviceSurfaceCapabilitiesKHR);
    MATCH(GetPhysicalDeviceSurfaceFormatsKHR);
    MATCH(GetPhysicalDeviceSurfacePresentModesKHR);
    MATCH(GetPhysicalDeviceSurfaceCapabilities2KHR);
    MATCH(GetPhysicalDeviceSurfaceFormats2KHR);
#undef MATCH
    return NULL;
}

static PFN_vkVoidFunction layer_intercept_device(const char *name) {
#define MATCH(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)layer_##n
    MATCH(GetDeviceProcAddr);
    MATCH(DestroyDevice);
    MATCH(CreateSwapchainKHR);
    MATCH(DestroySwapchainKHR);
    MATCH(GetSwapchainImagesKHR);
    MATCH(AcquireNextImageKHR);
    MATCH(QueuePresentKHR);
    MATCH(GetRefreshCycleDurationGOOGLE);
    MATCH(GetPastPresentationTimingGOOGLE);
    MATCH(QueueSubmit);
    MATCH(QueueWaitIdle);
    MATCH(DeviceWaitIdle);
    MATCH(WaitForFences);
    MATCH(ResetFences);
    MATCH(CmdPipelineBarrier);
    MATCH(EndCommandBuffer);
    MATCH(CreateRenderPass);
    MATCH(CreateRenderPass2);
    MATCH(CreateRenderPass2KHR);
#undef MATCH
    return NULL;
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
layer_GetDeviceProcAddr(VkDevice dev, const char *name) {
    DevNode *node = dev_lookup(dispatch_key(dev));
    if (!node) return NULL;
    if (node->passthrough) {
        /* Non-NVIDIA passthrough device.  Return the next layer's function
           for everything except the two functions we must still intercept:
           DestroyDevice (to clean up our DevNode) and CreateSwapchainKHR
           (to substitute the real ICD surface handle for our Surface*
           wrapper before forwarding to the ICD). */
        if (!strcmp(name, "vkDestroyDevice"))     return (PFN_vkVoidFunction)layer_DestroyDevice;
        if (!strcmp(name, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)layer_CreateSwapchainKHR;
        return node->d.GetDeviceProcAddr(dev, name);
    }
    PFN_vkVoidFunction fn = layer_intercept_device(name);
    if (fn) return fn;
    return node->d.GetDeviceProcAddr(dev, name);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
layer_GetInstanceProcAddr(VkInstance inst, const char *name) {
    PFN_vkVoidFunction fn = layer_intercept_instance(name);
    if (fn) return fn;
    fn = layer_intercept_device(name);
    if (fn) return fn;
    if (!inst) return NULL;
    InstNode *node = inst_lookup(dispatch_key(inst));
    if (!node) return NULL;
    return node->d.GetInstanceProcAddr(inst, name);
}

/* ----------------------------------------------------------------------- */
/* Loader negotiation (Vulkan layer interface v2)                          */
/* ----------------------------------------------------------------------- */

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pInterface) {
    if (pInterface->loaderLayerInterfaceVersion < 2) return VK_ERROR_INITIALIZATION_FAILED;
    pInterface->loaderLayerInterfaceVersion = 2;
    /* Wire negotiate to our INTERNAL (hidden) GIPA/GDPA, not the public
       wrappers. The public wrappers are no longer exported by symbol; this
       eliminates any chance of the loader's ICD-load path dlsym-ing our
       symbol and re-entering us. The loader doesn't need the public symbol
       in interface v2 — it stores these function pointers and calls them
       directly. */
    pInterface->pfnGetInstanceProcAddr      = layer_GetInstanceProcAddr;
    pInterface->pfnGetDeviceProcAddr        = layer_GetDeviceProcAddr;
    pInterface->pfnGetPhysicalDeviceProcAddr = NULL;
    layer_log_init();

    /* Loud one-time banner so a tester running an older cached binary can spot
       the version mismatch immediately. Bump when the layer's behaviour changes. */
    fprintf(stderr, "[" LAYER_NAME "] loaded, build %s %s (negotiated interface v2)\n",
            __DATE__, __TIME__);
    return VK_SUCCESS;
}
