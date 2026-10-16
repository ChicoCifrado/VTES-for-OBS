include(FetchContent)

set(OpenCV_VERSION "5.0.0")

# ── Force OpenCV to use its own 3rd-party libs ─────────────────────────
set(OPENCV_FORCE_3RDPARTY_BUILD ON CACHE BOOL "" FORCE)
set(OPENCV_DOWNLOAD_EXTERNAL_MODULES OFF CACHE BOOL "" FORCE)
set(OPENCV_ENABLE_NONFREE OFF CACHE BOOL "" FORCE)
set(BUILD_PROTOBUF ON CACHE BOOL "" FORCE)

# ── Static build, no shared ────────────────────────────────────────────
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# ── Disable tests, docs, apps, packages ────────────────────────────────
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(INSTALL_CREATE_DISTRIB OFF CACHE BOOL "" FORCE)
set(INSTALL_BIN_EXAMPLES OFF CACHE BOOL "" FORCE)
set(INSTALL_C_EXAMPLES OFF CACHE BOOL "" FORCE)
set(INSTALL_PYTHON_EXAMPLES OFF CACHE BOOL "" FORCE)
set(INSTALL_ANDROID_EXAMPLES OFF CACHE BOOL "" FORCE)
set(INSTALL_TO_MANGLED_PATHS OFF CACHE BOOL "" FORCE)
set(INSTALL_TESTS OFF CACHE BOOL "" FORCE)

# ── Disable language bindings ─────────────────────────────────────────
set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
set(PYTHON_EXECUTABLE "" CACHE FILEPATH "" FORCE)
set(PYTHON_INCLUDE_DIR "" CACHE FILEPATH "" FORCE)
set(PYTHON_LIBRARY "" CACHE FILEPATH "" FORCE)
set(BUILD_opencv_js OFF CACHE BOOL "" FORCE)
set(BUILD_ANDROID_PROJECTS OFF CACHE BOOL "" FORCE)
set(BUILD_ANDROID_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_ANDROID_SERVICE OFF CACHE BOOL "" FORCE)
set(BUILD_CUDA_STUBS OFF CACHE BOOL "" FORCE)
set(BUILD_FAT_JAVA_LIB OFF CACHE BOOL "" FORCE)
set(BUILD_OBJC OFF CACHE BOOL "" FORCE)

# ── Disable hardware features / backends ───────────────────────────────
set(WITH_1394 OFF CACHE BOOL "" FORCE)
set(WITH_ADE OFF CACHE BOOL "" FORCE)
set(WITH_ARAVIS OFF CACHE BOOL "" FORCE)
set(WITH_CLP OFF CACHE BOOL "" FORCE)
set(WITH_CUDA OFF CACHE BOOL "" FORCE)
set(WITH_CUFFT OFF CACHE BOOL "" FORCE)
set(WITH_CUBLAS OFF CACHE BOOL "" FORCE)
set(WITH_EIGEN OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
set(WITH_FLATBUFFERS OFF CACHE BOOL "" FORCE)
set(WITH_GDAL OFF CACHE BOOL "" FORCE)
set(WITH_GRAPHICSAPI OFF CACHE BOOL "" FORCE)
set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)
set(WITH_GTK OFF CACHE BOOL "" FORCE)
set(WITH_HALIDE OFF CACHE BOOL "" FORCE)
set(WITH_HPX OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_HDR OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_PFM OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_PXM OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_SUNRASTER OFF CACHE BOOL "" FORCE)
set(WITH_INF_ENGINE OFF CACHE BOOL "" FORCE)
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_ITT OFF CACHE BOOL "" FORCE)
set(WITH_JASPER OFF CACHE BOOL "" FORCE)
set(WITH_LAPACK OFF CACHE BOOL "" FORCE)
set(WITH_MATLAB OFF CACHE BOOL "" FORCE)
set(WITH_MSMF OFF CACHE BOOL "" FORCE)
set(WITH_NGRAPH OFF CACHE BOOL "" FORCE)
set(WITH_NVCUVID OFF CACHE BOOL "" FORCE)
set(WITH_OAK OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL OFF CACHE BOOL "" FORCE)
set(WITH_OPENCLAMDBLAS OFF CACHE BOOL "" FORCE)
set(WITH_OPENCLAMDFFT OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL_SVM OFF CACHE BOOL "" FORCE)
set(WITH_ONNXRUNTIME OFF CACHE BOOL "" FORCE)
set(DOWNLOAD_ONNXRUNTIME OFF CACHE BOOL "" FORCE)
set(DOWNLOAD_ONNXRUNTIME_GPU OFF CACHE BOOL "" FORCE)
set(OPENCV_DNN_CUDA OFF CACHE BOOL "" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
set(WITH_OPENGL OFF CACHE BOOL "" FORCE)
set(WITH_OPENJPEG OFF CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)
set(WITH_OPENNI OFF CACHE BOOL "" FORCE)
set(WITH_OPENNI2 OFF CACHE BOOL "" FORCE)
set(WITH_OPENVX OFF CACHE BOOL "" FORCE)
set(WITH_PTHREADS_PF OFF CACHE BOOL "" FORCE)
set(WITH_PVAPI OFF CACHE BOOL "" FORCE)
set(WITH_QT OFF CACHE BOOL "" FORCE)
set(WITH_QUIRC OFF CACHE BOOL "" FORCE)
set(WITH_TBB OFF CACHE BOOL "" FORCE)
set(WITH_TIFF OFF CACHE BOOL "" FORCE)
set(WITH_TIMVX OFF CACHE BOOL "" FORCE)
set(WITH_V4L OFF CACHE BOOL "" FORCE)
set(WITH_VA OFF CACHE BOOL "" FORCE)
set(WITH_VA_INTEL OFF CACHE BOOL "" FORCE)
set(WITH_VTK OFF CACHE BOOL "" FORCE)
set(WITH_VULKAN OFF CACHE BOOL "" FORCE)
set(WITH_WEBP OFF CACHE BOOL "" FORCE)
set(WITH_XIMEA OFF CACHE BOOL "" FORCE)

# ── Enable image codecs ────────────────────────────────────────────────
# JPEG for cv::imencode(".jpg", ...), PNG for cv::imread generic support
set(WITH_JPEG ON CACHE BOOL "" FORCE)
set(WITH_PNG ON CACHE BOOL "" FORCE)

# ── Enable only the modules we actually use ────────────────────────────
set(BUILD_opencv_core ON CACHE BOOL "" FORCE)
set(BUILD_opencv_imgproc ON CACHE BOOL "" FORCE)
set(BUILD_opencv_imgcodecs ON CACHE BOOL "" FORCE)
set(BUILD_opencv_video ON CACHE BOOL "" FORCE)       # cv::KalmanFilter
set(BUILD_opencv_geometry ON CACHE BOOL "" FORCE)     # getPerspectiveTransform, warpPerspective
set(BUILD_opencv_dnn ON CACHE BOOL "" FORCE)          # blobFromImage
set(BUILD_opencv_flann ON CACHE BOOL "" FORCE)        # required by geometry
set(BUILD_opencv_features ON CACHE BOOL "" FORCE)     # opencv2/features.hpp (umbrella)
set(BUILD_opencv_features2d ON CACHE BOOL "" FORCE)   # opencv2/features2d.hpp (umbrella)

# ── Disable every other module ─────────────────────────────────────────
set(BUILD_opencv_calib3d OFF CACHE BOOL "" FORCE)

set(BUILD_opencv_gapi OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_highgui OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_ml OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_objdetect OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_photo OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_stitching OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_videoio OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_calib OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_stereo OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_world OFF CACHE BOOL "" FORCE)

# ── Fetch and build OpenCV 5.0.0 from source ──────────────────────────
FetchContent_Declare(
  opencv
  GIT_REPOSITORY https://github.com/opencv/opencv.git
  GIT_TAG ${OpenCV_VERSION}
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

FetchContent_GetProperties(opencv)
if(NOT opencv_POPULATED)
  FetchContent_Populate(opencv)

  # ── Patch GetTempPathW → GetTempPathA for modern WinSDK (UNICODE conflict) ─
  set(_fs_cpp "${opencv_SOURCE_DIR}/modules/core/src/utils/filesystem.cpp")
  file(READ "${_fs_cpp}" _fs_text)
  string(REPLACE "GetTempPath(" "GetTempPathA(" _fs_text "${_fs_text}")
  file(WRITE "${_fs_cpp}" "${_fs_text}")
  unset(_fs_cpp)
  unset(_fs_text)

  # ── Patch EnableProfiling(char*) → EnableProfiling(wchar_t*) for ORT 1.25+ ─
  set(_ort_cpp "${opencv_SOURCE_DIR}/modules/dnn/src/net_impl_backend.cpp")
  file(READ "${_ort_cpp}" _ort_text)
  string(REPLACE
    "opts.EnableProfiling(ort_profile_path_prefix.c_str());"
    "#ifdef _WIN32\n        std::wstring wprofile(ort_profile_path_prefix.begin(), ort_profile_path_prefix.end());\n        opts.EnableProfiling(wprofile.c_str());\n#else\n        opts.EnableProfiling(ort_profile_path_prefix.c_str());\n#endif"
    _ort_text "${_ort_text}")
  file(WRITE "${_ort_cpp}" "${_ort_text}")
  unset(_ort_cpp)
  unset(_ort_text)

  # Force dynamic CRT (/MD) to match the rest of the project.
  # OpenCV sets BUILD_WITH_STATIC_CRT=ON by default for static builds,
  # which overrides CMAKE_MSVC_RUNTIME_LIBRARY to "MultiThreaded".
  if(MSVC)
    set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "" FORCE)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "" FORCE)
  endif()

  # Disable Unicode font download — the download/copy logic in OpenCV 5.0.0
  # fails when the build tree is freshly populated (dest dir doesn't exist yet).
  # Only affects cv::putText with Unicode strings (we only need ASCII/Hershey).
  set(WITH_UNIFONT OFF CACHE BOOL "" FORCE)

  # ── Disable MLAS on MSVC, fix paths on other platforms ──
  # MLAS (Microsoft Linear Algebra Subprograms) is an ASM-optimized SGEMM.
  # On MSVC the .S assembly files can't compile (no GNU ASM in MSBuild),
  # and the ASM compiler found (MinGW GCC via PATH) produces .obj files
  # that MSBuild's linker can't find. Early-return disables MLAS; DNN
  # falls back to its built-in SGEMM (fine for blobFromImage only).
  # On Unix: patch the path references (CMAKE_SOURCE_DIR → CMAKE_CURRENT_SOURCE_DIR)
  # since OpenCV is a subdirectory.
  set(_mlas_cmake "${opencv_SOURCE_DIR}/3rdparty/mlas/CMakeLists.txt")
  file(READ "${_mlas_cmake}" _mlas_text)
  if(MSVC)
    set(_mlas_text "if(MSVC)\n  return()\nendif()\n${_mlas_text}")
  else()
    string(REPLACE
      "\${CMAKE_SOURCE_DIR}/modules/dnn/src/layers/cpu_kernels/mlas_threading.cpp"
      "\${CMAKE_CURRENT_SOURCE_DIR}/../../modules/dnn/src/layers/cpu_kernels/mlas_threading.cpp"
      _mlas_text "${_mlas_text}")
    string(REPLACE
      "\${CMAKE_SOURCE_DIR}/modules/core/include"
      "\${CMAKE_CURRENT_SOURCE_DIR}/../../modules/core/include"
      _mlas_text "${_mlas_text}")
  endif()
  file(WRITE "${_mlas_cmake}" "${_mlas_text}")
  unset(_mlas_cmake)
  unset(_mlas_text)

  add_subdirectory(${opencv_SOURCE_DIR} ${opencv_BINARY_DIR})

  # On MSVC, strip "pthread" (including $<LINK_ONLY:pthread> generator
  # expressions) from OpenCV module INTERFACE_LINK_LIBRARIES so the MSVC
  # linker doesn't fail with LNK1181 (pthread.lib doesn't exist on Windows).
  if(MSVC)
    foreach(_tgt opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn
                 opencv_video opencv_geometry opencv_flann opencv_features
                 opencv_features2d opencv_ptcloud)
      if(TARGET ${_tgt})
        get_target_property(_libs ${_tgt} INTERFACE_LINK_LIBRARIES)
        if(_libs)
          set(_before "${_libs}")
          list(FILTER _libs EXCLUDE REGEX "LINK_ONLY:pthread")
          if("pthread" IN_LIST _libs)
            list(REMOVE_ITEM _libs "pthread")
          endif()
          if(NOT "${_libs}" STREQUAL "${_before}")
            set_target_properties(${_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_libs}")
          endif()
        endif()
      endif()
    endforeach()
    unset(_tgt)
    unset(_libs)
    unset(_before)
  endif()

endif()

# ── Wrap OpenCV CMake targets as the "OpenCV" interface the parent expects ──
if(NOT TARGET OpenCV)
  add_library(OpenCV INTERFACE)
  target_link_libraries(OpenCV INTERFACE
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_video
    opencv_geometry
    opencv_dnn
    opencv_flann
  )
  target_include_directories(OpenCV SYSTEM INTERFACE
    "${opencv_SOURCE_DIR}/include"
    "${opencv_SOURCE_DIR}/modules/core/include"
    "${opencv_SOURCE_DIR}/modules/imgproc/include"
    "${opencv_SOURCE_DIR}/modules/imgcodecs/include"
    "${opencv_SOURCE_DIR}/modules/video/include"
    "${opencv_SOURCE_DIR}/modules/geometry/include"
    "${opencv_SOURCE_DIR}/modules/dnn/include"
    "${opencv_SOURCE_DIR}/modules/flann/include"
    "${opencv_SOURCE_DIR}/modules/features/include"
    "${opencv_SOURCE_DIR}/modules/features2d/include"
    "${opencv_BINARY_DIR}"
  )
endif()
