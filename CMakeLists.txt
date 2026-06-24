cmake_minimum_required(VERSION 3.18...3.31)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

# OpenCV 5 requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Suppress CMP0175 warning (PRE_BUILD deprecated in add_custom_command)
if(POLICY CMP0175)
  cmake_policy(SET CMP0175 NEW)
endif()

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_qt(COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt::Core Qt::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME} PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header
                                  -Wno-comma>)
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON
               AUTOUIC ON
               AUTORCC ON)
endif()

set(USE_SYSTEM_OPENCV
    OFF
    CACHE STRING "Use system OpenCV")
if(USE_SYSTEM_OPENCV)
  # Use find_package without COMPONENTS for world builds (both OpenCV 4 and 5)
  # OpenCV_LIBS will contain the correct library (opencv_world4xx or opencv_world5xx)
  find_package(OpenCV REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${OpenCV_LIBS}")
  target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PUBLIC "${OpenCV_INCLUDE_DIRS}")
  # Install OpenCV DLL alongside the plugin for distribution
  if(WIN32)
    get_filename_component(_cv_root "${OpenCV_DIR}" DIRECTORY)
    file(GLOB_RECURSE _cv_dlls "${_cv_root}/opencv_world*.dll")
    list(SORT _cv_dlls)
    if(_cv_dlls)
      install(FILES ${_cv_dlls} DESTINATION "obs-plugins/64bit")
      message(STATUS "OpenCV DLL for installer: ${_cv_dlls}")
    else()
      message(WARNING "OpenCV DLL not found under ${_cv_root} — copy manually after install")
    endif()
    unset(_cv_root)
    unset(_cv_dlls)
  endif()
else()
  include(cmake/FetchOpenCV.cmake)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OpenCV)
endif()

# add the vendor and src folders to the include path
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE vendor)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE src)

# ─── ONNX Runtime (for YOLO model — pinned version for CUDA/cuDNN compatibility) ─
# Uses a specific ORT release from GitHub instead of OpenCV's auto-downloaded one
# to ensure reproducible builds and cuDNN version compatibility.
set(ORT_VERSION "1.23.1")
set(ORT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/onnxruntime-gpu-${ORT_VERSION}")
set(ORT_DOWNLOAD_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-gpu-${ORT_VERSION}.zip")
set(ORT_ZIP "${CMAKE_CURRENT_BINARY_DIR}/_deps/onnxruntime-gpu-${ORT_VERSION}.zip")

if(NOT EXISTS "${ORT_DIR}/lib/onnxruntime.lib")
    message(STATUS "Downloading ORT ${ORT_VERSION} (CUDA 12, cuDNN 9)...")
    file(DOWNLOAD "${ORT_DOWNLOAD_URL}" "${ORT_ZIP}" INACTIVITY_TIMEOUT 60 SHOW_PROGRESS)
    file(ARCHIVE_EXTRACT INPUT "${ORT_ZIP}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/_deps/")
    # The zip contains a top-level folder like onnxruntime-win-x64-gpu-1.20.1/
    file(GLOB _ort_extracted "${CMAKE_CURRENT_BINARY_DIR}/_deps/onnxruntime-win-x64-gpu-*")
    list(SORT _ort_extracted)
    list(GET _ort_extracted -1 _ort_extracted_dir)
    file(RENAME "${_ort_extracted_dir}" "${ORT_DIR}")
endif()

if(EXISTS "${ORT_DIR}/lib/onnxruntime.lib")
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${ORT_DIR}/include")
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${ORT_DIR}/lib/onnxruntime.lib")
    message(STATUS "ORT ${ORT_VERSION} found: ${ORT_DIR}")
    # Install ORT DLLs alongside the plugin
    install(FILES "${ORT_DIR}/lib/onnxruntime.dll"
                  "${ORT_DIR}/lib/onnxruntime_providers_shared.dll"
            DESTINATION "obs-plugins/64bit")
    if(EXISTS "${ORT_DIR}/lib/onnxruntime_providers_cuda.dll")
        install(FILES "${ORT_DIR}/lib/onnxruntime_providers_cuda.dll"
                DESTINATION "obs-plugins/64bit")
    endif()
    if(EXISTS "${ORT_DIR}/lib/onnxruntime_providers_dml.dll")
        install(FILES "${ORT_DIR}/lib/onnxruntime_providers_dml.dll"
                DESTINATION "obs-plugins/64bit")
    endif()
    # Install CUDA runtime DLLs needed by the CUDA provider
    # Convert Windows backslash paths to CMake forward-slash paths
    set(_cuda_dll_dirs "")
    if(DEFINED ENV{CUDA_PATH})
        set(_cuda_path "$ENV{CUDA_PATH}")
        file(TO_CMAKE_PATH "${_cuda_path}" _cuda_path)
        list(APPEND _cuda_dll_dirs "${_cuda_path}/bin")
    endif()
    foreach(_dll cudart64_12 cublas64_12 cublasLt64_12 cufft64_11)
        set(_found FALSE)
        foreach(_dir ${_cuda_dll_dirs})
            if(EXISTS "${_dir}/${_dll}.dll")
                install(FILES "${_dir}/${_dll}.dll" DESTINATION "obs-plugins/64bit")
                message(STATUS "CUDA DLL found: ${_dir}/${_dll}.dll")
                set(_found TRUE)
                break()
            endif()
        endforeach()
        if(NOT _found)
            message(STATUS "CUDA DLL NOT found (will be loaded from PATH if available): ${_dll}.dll")
        endif()
    endforeach()
    unset(_cuda_dll_dirs)
    unset(_cuda_path)
else()
    message(FATAL_ERROR "ORT ${ORT_VERSION} not found at ${ORT_DIR}")
endif()
unset(ORT_VERSION)
unset(ORT_DIR)

# ── TensorRT (for GPU inference via tensor cores, replaces CUDA EP) ──
set(TENSORRT_FOUND FALSE)
if(DEFINED ENV{TENSORRT_ROOT})
    set(_trt_root "$ENV{TENSORRT_ROOT}")
    file(TO_CMAKE_PATH "${_trt_root}" _trt_root)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS "${_trt_root}/include" NO_DEFAULT_PATH)
    find_library(TENSORRT_LIB nvinfer_lean_11 PATHS "${_trt_root}/lib" NO_DEFAULT_PATH)
    find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin_11 PATHS "${_trt_root}/lib" NO_DEFAULT_PATH)
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
        set(TENSORRT_FOUND TRUE)
        target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${TENSORRT_INCLUDE_DIR}")
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${TENSORRT_LIB}")
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE TENSORRT_AVAILABLE)
        if(TENSORRT_PLUGIN_LIB)
            target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${TENSORRT_PLUGIN_LIB}")
        endif()
        # Deploy only inference DLLs; skip the 400MB builder resource DLLs
        foreach(_trt_dll_suffix "nvinfer_lean_11.dll" "nvinfer_plugin_11.dll")
            find_file(_trt_dll "${_trt_dll_suffix}" PATHS "${_trt_root}/bin" NO_DEFAULT_PATH)
            if(_trt_dll)
                install(FILES "${_trt_dll}" DESTINATION "obs-plugins/64bit")
                message(STATUS "TensorRT DLL: ${_trt_dll}")
                unset(_trt_dll)
            else()
                find_file(_trt_dll "${_trt_dll_suffix}" PATHS "${_trt_root}/lib" NO_DEFAULT_PATH)
                if(_trt_dll)
                    install(FILES "${_trt_dll}" DESTINATION "obs-plugins/64bit")
                    message(STATUS "TensorRT DLL (from lib): ${_trt_dll}")
                    unset(_trt_dll)
                endif()
            endif()
        endforeach()
        unset(_trt_dll_suffix)
        # Add tensorrt_detector.cpp to sources
        target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/detection/tensorrt_detector.cpp)
        # CUDA include path + cudart.lib (for cuda_runtime.h and CUDA runtime symbols)
        if(DEFINED ENV{CUDA_PATH})
            set(_cuda_inc "$ENV{CUDA_PATH}")
            file(TO_CMAKE_PATH "${_cuda_inc}" _cuda_inc)
            target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${_cuda_inc}/include")
            find_library(CUDART_LIB cudart PATHS "${_cuda_inc}/lib/x64" NO_DEFAULT_PATH)
            if(CUDART_LIB)
                target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${CUDART_LIB}")
                unset(CUDART_LIB)
            endif()
            unset(_cuda_inc)
        endif()
        message(STATUS "TensorRT found: ${_trt_root}")
    else()
        message(WARNING "TensorRT includes/libs not found under ${_trt_root}. GPU inference disabled.")
    endif()
    unset(_trt_root)
else()
    message(STATUS "TENSORRT_ROOT not set — GPU inference via TensorRT disabled. Set TENSORRT_ROOT to enable.")
endif()
unset(TENSORRT_INCLUDE_DIR)
unset(TENSORRT_LIB)
unset(TENSORRT_PLUGIN_LIB)

# ── CUDA kernels (pre-compiled PTX, loaded at runtime via Driver API) ──
# The PTX is generated offline on a Linux dev machine and checked into
# the repo.  At runtime vtes_cuda_kernels_ptx.h is compiled as a C string
# and loaded via cuModuleLoadData.  No nvcc invocation during build.
if(DEFINED ENV{CUDA_PATH})
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        # Host wrappers compiled by MSVC (no CUDA keywords)
        target_sources(${CMAKE_PROJECT_NAME} PRIVATE
            src/ocr/nis_upscaler.cpp
            src/detection/vampire_classifier.cpp
            src/cuda/cuda_kernel_launcher.cpp)

        target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::cudart)

        # CUDA Driver API (cuModuleLoadData, cuLaunchKernel, etc.)
        find_library(CUDA_DRIVER_LIB NAMES cuda PATHS ${CUDAToolkit_LIBRARY_DIR})
        if(CUDA_DRIVER_LIB)
            target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${CUDA_DRIVER_LIB})
            target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAVE_CUDA_NIS)
            message(STATUS "CUDA Driver API found at ${CUDA_DRIVER_LIB}")
        else()
            message(WARNING "CUDA Driver library (cuda.lib / libcuda.so) not found — CUDA PTX kernels disabled")
        endif()

        message(STATUS "CUDA PTX kernel loader enabled (CUDA ${CUDAToolkit_VERSION})")
    else()
        message(WARNING "CUDA_PATH set but CUDAToolkit not found — CUDA kernels disabled")
    endif()
else()
    message(STATUS "CUDA_PATH not set — CUDA kernels disabled")
endif()

# cpp-httplib needs WinSock on Windows; NO OpenSSL (leave CPPHTTPLIB_OPENSSL_SUPPORT undefined)
if(WIN32)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ws2_32)
endif()

# libcurl for OCR API lookup
find_package(CURL REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CURL::libcurl)

# Link plugin-support for version constants and include propagation
if(TARGET plugin-support)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plugin-support)
endif()

# ─── Tesseract OCR (loaded dynamically at runtime, no .lib needed) ────

# ===== OpenCV DNN backend (ONNX models via cv::dnn::Net) =====
target_sources(
  ${CMAKE_PROJECT_NAME}
  PRIVATE src/plugin-main.c
          src/detect-filter-obb.cpp
          src/detect-filter-obb-info.c
          src/detect-filter-utils.cpp
          src/obs-utils/obs-utils.cpp
          src/detection/yolo_detector.cpp
          src/detection/contour_detector.cpp
          src/sort/Sort.cpp
          src/classifier/vtes_card_classifier.cpp
          src/ocr/vtes_ocr.cpp
          src/ocr/vtes_api_lookup.cpp)

# Configure plugin target properties (install, resources, etc.)
set_target_properties_plugin(${CMAKE_PROJECT_NAME})
