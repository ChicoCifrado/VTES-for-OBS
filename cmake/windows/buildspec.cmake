# CMake Windows build dependencies module

include_guard(GLOBAL)

include(buildspec_common)

# _check_dependencies_windows: Set up Windows slice for _check_dependencies
function(_check_dependencies_windows)
  set(arch ${CMAKE_GENERATOR_PLATFORM})
  set(platform windows-${arch})

  set(dependencies_dir "${CMAKE_CURRENT_SOURCE_DIR}/.deps")

  # SDK local del OBS instalado (lo genera scripts/make_obs_sdk.bat). Cuando OBS_SDK_DIR apunta a un SDK valido, el
  # plugin se compila y enlaza contra el OBS del sistema y no se descarga ni compila libobs desde fuente. De las
  # dependencias descargadas solo sigue haciendo falta obs-deps, que aporta libcurl (API de OCR); si falta, se vuelve
  # al camino normal entero.
  if(OBS_SDK_DIR)
    file(TO_CMAKE_PATH "${OBS_SDK_DIR}" _obs_sdk_dir)
    # cmake-format: off
    string(JSON _prebuilt_version GET ${buildspec} dependencies prebuilt version)
    # cmake-format: on
    set(_prebuilt_dir "${dependencies_dir}/obs-deps-${_prebuilt_version}-${arch}")
    if(EXISTS "${_obs_sdk_dir}/cmake/libobsConfig.cmake"
       AND EXISTS "${_prebuilt_dir}/share/obs-deps/VERSION")
      message(STATUS "Using local OBS SDK from OBS_SDK_DIR: ${_obs_sdk_dir}")
      list(PREPEND CMAKE_PREFIX_PATH "${_obs_sdk_dir}")
      list(APPEND CMAKE_PREFIX_PATH "${_prebuilt_dir}")
      list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
      # cmake-format: off
      set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} CACHE PATH "CMake prefix search path" FORCE)
      # cmake-format: on
      return()
    endif()
    message(STATUS "OBS_SDK_DIR is set but is not a usable OBS SDK; using downloaded dependencies")
  endif()

  set(prebuilt_filename "windows-deps-VERSION-ARCH-REVISION.zip")
  set(prebuilt_destination "obs-deps-VERSION-ARCH")
  set(qt6_filename "windows-deps-qt6-VERSION-ARCH-REVISION.zip")
  set(qt6_destination "obs-deps-qt6-VERSION-ARCH")
  set(obs-studio_filename "VERSION.zip")
  set(obs-studio_destination "obs-studio-VERSION")
  set(dependencies_list prebuilt qt6 obs-studio)

  _check_dependencies()
endfunction()

_check_dependencies_windows()
