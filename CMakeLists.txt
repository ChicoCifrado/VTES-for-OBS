cmake_minimum_required(VERSION 3.16...3.26)

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

# ─── ONNX Runtime (for YOLO model — OpenCV DNN parser doesn't support dynamic reshape) ─
if(NOT USE_SYSTEM_OPENCV AND DEFINED opencv_BINARY_DIR)
    set(_ort_root "${opencv_BINARY_DIR}/3rdparty/onnxruntime")
    file(GLOB _ort_dirs "${_ort_root}/*")
    if(_ort_dirs)
        list(SORT _ort_dirs)
        list(GET _ort_dirs -1 _ort_dir)
        target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${_ort_dir}/include")
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${_ort_dir}/lib/onnxruntime.lib")
        message(STATUS "ORT found: ${_ort_dir}")
        # Install ORT DLLs alongside the plugin
        install(FILES "${_ort_dir}/lib/onnxruntime.dll"
                      "${_ort_dir}/lib/onnxruntime_providers_shared.dll"
                DESTINATION "obs-plugins/64bit")
        if(EXISTS "${_ort_dir}/lib/onnxruntime_providers_cuda.dll")
            install(FILES "${_ort_dir}/lib/onnxruntime_providers_cuda.dll"
                    DESTINATION "obs-plugins/64bit")
        endif()
    else()
        message(WARNING "ORT directory not found under ${_ort_root}")
    endif()
    unset(_ort_root)
    unset(_ort_dirs)
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

# ─── Optional Tesseract OCR for card name reading ──────────────────────
option(USE_SYSTEM_TESSERACT "Use system Tesseract (vcpkg/chocolatey)" OFF)
if(USE_SYSTEM_TESSERACT)
  find_package(Tesseract QUIET)
  if(Tesseract_FOUND)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VTES_HAVE_TESSERACT)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Tesseract::libtesseract)
    message(STATUS "Tesseract OCR enabled: ${Tesseract_DIR}")
  else()
    message(STATUS "Tesseract not found — OCR fallback disabled")
  endif()
endif()

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
