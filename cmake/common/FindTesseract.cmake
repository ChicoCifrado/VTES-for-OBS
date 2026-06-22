# FindTesseract.cmake
# Finds Tesseract OCR on Windows (standard install) and other platforms.
#
# Sets:
#   Tesseract_FOUND
#   Tesseract_INCLUDE_DIRS
#   Tesseract_LIBRARY
#   Tesseract::libtesseract (imported target)

if(NOT WIN32)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_Tesseract QUIET tesseract)
    endif()
endif()

if(WIN32)
    set(_TESSERACT_PATHS
        "C:/Program Files/Tesseract-OCR"
        "C:/Program Files (x86)/Tesseract-OCR"
        "$ENV{Tesseract_ROOT}"
        "$ENV{TESSERACT_ROOT}"
    )
endif()

find_path(Tesseract_INCLUDE_DIR
    NAMES tesseract/capi.h
    HINTS
        ${PC_Tesseract_INCLUDE_DIRS}
        ${_TESSERACT_PATHS}
    PATH_SUFFIXES
        include
)

find_library(Tesseract_LIBRARY
    NAMES
        tesseract
        libtesseract
        tesseract50
        tesseract41
    HINTS
        ${PC_Tesseract_LIBRARY_DIRS}
        ${_TESSERACT_PATHS}
    PATH_SUFFIXES
        lib
        lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tesseract
    REQUIRED_VARS
        Tesseract_LIBRARY
        Tesseract_INCLUDE_DIR
)

if(Tesseract_FOUND)
    set(Tesseract_INCLUDE_DIRS ${Tesseract_INCLUDE_DIR})

    if(NOT TARGET Tesseract::libtesseract)
        add_library(Tesseract::libtesseract UNKNOWN IMPORTED)
        set_target_properties(Tesseract::libtesseract PROPERTIES
            IMPORTED_LOCATION "${Tesseract_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${Tesseract_INCLUDE_DIRS}"
        )
    endif()

    mark_as_advanced(
        Tesseract_INCLUDE_DIR
        Tesseract_LIBRARY
    )
endif()
