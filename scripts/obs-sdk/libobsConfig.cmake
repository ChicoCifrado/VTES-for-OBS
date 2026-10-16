# Config de libobs para el SDK local del OBS instalado (ver make_obs_sdk.bat).
#
# A diferencia del libobsConfig.cmake que genera una compilacion de obs-studio
# desde fuente, este describe el runtime que ya esta instalado en el sistema:
# obs.lib se extrajo de los exports del obs.dll instalado y las cabeceras son
# las de esa misma version. El SDK vive en .deps/obs-sdk y se activa con
# OBS_SDK_DIR (lo fija CMakePresets.json); si el directorio no existe, el build
# descarga las dependencias de buildspec.json como siempre.

include(CMakeFindDependencyMacro)
find_dependency(Threads)

get_filename_component(_libobs_sdk_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# libobs declara w32-pthreads como dependencia publica en Windows. Este plugin
# no usa pthread_* directamente, pero el target debe existir para que la cadena
# de enlace sea la misma que con un libobs oficial.
if(NOT TARGET OBS::w32-pthreads)
  add_library(OBS::w32-pthreads SHARED IMPORTED)
  set_target_properties(OBS::w32-pthreads PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_libobs_sdk_dir}/include"
    IMPORTED_IMPLIB "${_libobs_sdk_dir}/lib/w32-pthreads.lib"
    IMPORTED_LOCATION "${_libobs_sdk_dir}/bin/64bit/w32-pthreads.dll")
endif()

if(NOT TARGET OBS::libobs)
  add_library(OBS::libobs SHARED IMPORTED)
  set_target_properties(OBS::libobs PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "HAVE_OBSCONFIG_H"
    INTERFACE_INCLUDE_DIRECTORIES "${_libobs_sdk_dir}/include"
    INTERFACE_LINK_LIBRARIES "Threads::Threads;OBS::w32-pthreads"
    IMPORTED_IMPLIB "${_libobs_sdk_dir}/lib/obs.lib"
    IMPORTED_LOCATION "${_libobs_sdk_dir}/bin/64bit/obs.dll")
endif()

unset(_libobs_sdk_dir)
