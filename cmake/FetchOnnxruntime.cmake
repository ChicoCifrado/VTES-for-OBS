include(FetchContent)

set(CUSTOM_ONNXRUNTIME_URL
    ""
    CACHE STRING "URL of a downloaded ONNX Runtime tarball")

set(CUSTOM_ONNXRUNTIME_HASH
    ""
    CACHE STRING "Hash of a downloaded ONNX Runtime tarball")

set(Onnxruntime_VERSION "1.25.0")

if(CUSTOM_ONNXRUNTIME_URL STREQUAL "")
  set(USE_PREDEFINED_ONNXRUNTIME ON)
else()
  if(CUSTOM_ONNXRUNTIME_HASH STREQUAL "")
    message(FATAL_ERROR "Both of CUSTOM_ONNXRUNTIME_URL and CUSTOM_ONNXRUNTIME_HASH must be present!")
  else()
    set(USE_PREDEFINED_ONNXRUNTIME OFF)
  endif()
endif()

if(USE_PREDEFINED_ONNXRUNTIME)
  set(Onnxruntime_BASEURL "https://github.com/microsoft/onnxruntime/releases/download/v${Onnxruntime_VERSION}")
  # Use official Microsoft release for Windows (includes static libs and DLLs)
  set(Onnxruntime_WINDOWS_VERSION "v${Onnxruntime_VERSION}")
  set(Onnxruntime_WINDOWS_BASEURL
      "https://github.com/microsoft/onnxruntime/releases/download/${Onnxruntime_WINDOWS_VERSION}")

  if(APPLE)
    set(Onnxruntime_URL "${Onnxruntime_BASEURL}/onnxruntime-osx-universal2-${Onnxruntime_VERSION}.tgz")
    set(Onnxruntime_HASH SHA256=9FA57FA6F202A373599377EF75064AE568FDA8DA838632B26A86024C7378D306)
  elseif(MSVC)
    # Standard Windows CPU package — DML provider is in onnxruntime_providers_shared.dll
    # (DirectML.dll ships with Windows 10 1903+ and is not bundled here)
    set(Onnxruntime_URL "${Onnxruntime_WINDOWS_BASEURL}/onnxruntime-win-x64-${Onnxruntime_VERSION}.zip")
  else()
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
      set(Onnxruntime_URL "${Onnxruntime_BASEURL}/onnxruntime-linux-aarch64-${Onnxruntime_VERSION}.tgz")
      set(Onnxruntime_HASH SHA256=70B6F536BB7AB5961D128E9DBD192368AC1513BFFB74FE92F97AAC342FBD0AC1)
    else()
      set(Onnxruntime_URL "${Onnxruntime_BASEURL}/onnxruntime-linux-x64-gpu-${Onnxruntime_VERSION}.tgz")
      set(Onnxruntime_HASH SHA256=613C53745EA4960ED368F6B3AB673558BB8561C84A8FA781B4EA7FB4A4340BE4)
    endif()
  endif()
else()
  set(Onnxruntime_URL "${CUSTOM_ONNXRUNTIME_URL}")
  set(Onnxruntime_HASH "${CUSTOM_ONNXRUNTIME_HASH}")
endif()

if(Onnxruntime_HASH)
  FetchContent_Declare(
    onnxruntime
    URL ${Onnxruntime_URL}
    URL_HASH ${Onnxruntime_HASH})
else()
  FetchContent_Declare(
    onnxruntime
    URL ${Onnxruntime_URL})
endif()
FetchContent_MakeAvailable(onnxruntime)

if(APPLE)
  set(Onnxruntime_LIB "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.${Onnxruntime_VERSION}.dylib")
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${Onnxruntime_LIB}")
  target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PUBLIC "${onnxruntime_SOURCE_DIR}/include")
  target_sources(${CMAKE_PROJECT_NAME} PRIVATE "${Onnxruntime_LIB}")
  set_property(SOURCE "${Onnxruntime_LIB}" PROPERTY MACOSX_PACKAGE_LOCATION Frameworks)
  source_group("Frameworks" FILES "${Onnxruntime_LIB}")
  # add a codesigning step
  add_custom_command(
    TARGET "${CMAKE_PROJECT_NAME}"
    PRE_BUILD
    COMMAND /usr/bin/codesign --force --verify --verbose --sign "${CODESIGN_IDENTITY}" "${Onnxruntime_LIB}")
  add_custom_command(
    TARGET "${CMAKE_PROJECT_NAME}"
    POST_BUILD
    COMMAND
      ${CMAKE_INSTALL_NAME_TOOL} -change "@rpath/libonnxruntime.${Onnxruntime_VERSION}.dylib"
      "@loader_path/../Frameworks/libonnxruntime.${Onnxruntime_VERSION}.dylib" $<TARGET_FILE:${CMAKE_PROJECT_NAME}>)
elseif(MSVC)
  # Official Microsoft release provides onnxruntime.lib (import library) + DLLs
  # Not the individual static libs like onnxruntime_session.lib
  add_library(Ort INTERFACE)
  
  # Main ONNX Runtime import library
  add_library(Ort::onnxruntime STATIC IMPORTED)
  set_target_properties(Ort::onnxruntime PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
    INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include)
  target_link_libraries(Ort INTERFACE Ort::onnxruntime)
  
  # External dependencies (if available)
  set(Onnxruntime_EXTERNAL_LIB_NAMES
      onnx;onnx_proto;libprotobuf-lite;re2;absl_throw_delegate;absl_hash;absl_city;absl_low_level_hash;absl_raw_hash_set)
  foreach(lib_name IN LISTS Onnxruntime_EXTERNAL_LIB_NAMES)
    if(EXISTS "${onnxruntime_SOURCE_DIR}/lib/${lib_name}.lib")
      add_library(Ort::${lib_name} STATIC IMPORTED)
      set_target_properties(Ort::${lib_name} PROPERTIES
        IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/${lib_name}.lib
        INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include)
      target_link_libraries(Ort INTERFACE Ort::${lib_name})
    endif()
  endforeach()

  # DirectML.dll ships with Windows 10 1903+ — not bundled in the CPU package

  # Install ONNX Runtime DLLs (main runtime and providers)
  # These are in lib/ for the official Microsoft Windows release
  if(EXISTS "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll")
    install(FILES "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll" DESTINATION "obs-plugins/64bit")
  endif()
  if(EXISTS "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll")
    install(FILES "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll" DESTINATION "obs-plugins/64bit")
  endif()

  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Ort)
else()
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(Onnxruntime_LINK_LIBS "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so.${Onnxruntime_VERSION}")
    set(Onnxruntime_INSTALL_LIBS ${Onnxruntime_LINK_LIBS})
  else()
    set(Onnxruntime_LINK_LIBS "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so.${Onnxruntime_VERSION}")
    set(Onnxruntime_INSTALL_LIBS ${Onnxruntime_LINK_LIBS}
                                 "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_shared.so")
    # TODO add other providers "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_cuda.so"
    # "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_tensorrt.so"
  endif()
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${Onnxruntime_LINK_LIBS})
  target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PUBLIC "${onnxruntime_SOURCE_DIR}/include")
  install(FILES ${Onnxruntime_INSTALL_LIBS} DESTINATION "${CMAKE_INSTALL_LIBDIR}/obs-plugins/${CMAKE_PROJECT_NAME}")
  set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_PROJECT_NAME}")
endif()