# CMake Windows defaults module

include_guard(GLOBAL)

# Enable find_package targets to become globally available targets
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)

include(buildspec)

# ── CPack packaging: NSIS installer ──────────────────────────────────
# Run: cpack -G NSIS -C RelWithDebInfo (from build directory)
# Requires NSIS: https://nsis.sourceforge.io/Download

# Find the NSIS compiler at configure time so CPack knows where it is
# (standard install paths; find_program also searches PATH)
find_program(NSIS_MAKENSIS makensis.exe PATHS
  "C:/Program Files (x86)/NSIS"
  "C:/Program Files/NSIS")
if(NSIS_MAKENSIS)
  set(CPACK_NSIS_EXECUTABLE "${NSIS_MAKENSIS}")
  message(STATUS "NSIS found: ${NSIS_MAKENSIS}")
else()
  message(WARNING "NSIS (makensis.exe) not found — run 'cpack -G NSIS' will fail")
endif()

set(CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${CMAKE_PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "VTES Card Scanner - Real-time card identification plugin for OBS Studio")
set(CPACK_PACKAGE_VENDOR "${PLUGIN_AUTHOR}")
set(CPACK_PACKAGE_CONTACT "${PLUGIN_EMAIL}")
set(CPACK_PACKAGE_HOMEPAGE_URL "${PLUGIN_WEBSITE}")

# Default install path -> C:\Program Files\obs-studio
# (so obs-plugins/64bit/ and data/obs-plugins/ resolve correctly)
set(CPACK_PACKAGE_INSTALL_DIRECTORY "obs-studio")
set(CPACK_NSIS_DISPLAY_NAME "VTES Card Scanner OBS Plugin")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
set(CPACK_NSIS_MODIFY_PATH OFF)
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
set(CPACK_NSIS_MANIFEST_DPI_AWARE ON)
set(CPACK_NSIS_CREATE_ICONS OFF)

# Output filename: vtes-card-scanner-1.0.0-Windows-x64.exe
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-Windows-x64")

# Output to release/ alongside build artifacts
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/release")

set(CPACK_GENERATOR "NSIS")

include(CPack)
