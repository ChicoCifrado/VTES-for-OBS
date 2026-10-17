@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

rem ===========================================================================
rem  make_obs_sdk.bat - SDK local de libobs a partir del OBS instalado
rem
rem  El bootstrap de CMake (cmake/windows/buildspec.cmake) descarga obs-studio
rem  y compila libobs desde fuente. Este script construye una alternativa
rem  ligera enlazando contra el OBS que ya esta instalado:
rem
rem    .deps\obs-sdk\
rem      include\          cabeceras de la version instalada
rem      lib\obs.lib       import lib generada desde los exports de obs.dll
rem      lib\w32-pthreads.lib
rem      bin\64bit\*.dll   copias de los DLL del OBS instalado
rem      cmake\            libobsConfig.cmake (+ version) que define OBS::libobs
rem
rem  CMake lo usa cuando OBS_SDK_DIR apunta al SDK (CMakePresets.json lo fija
rem  en .deps/obs-sdk). Si el directorio no existe, el build cae de vuelta a
rem  las dependencias de buildspec.json como siempre.
rem
rem  Uso: make_obs_sdk.bat [ruta_de_obs] [ruta_de_cabeceras]
rem    ruta_de_obs         por defecto C:\Program Files\obs-studio
rem    ruta_de_cabeceras   por defecto <repo>\.deps\include-32.2.2
rem                        (cabeceras de libobs de la MISMA version que el OBS
rem                        instalado; salen del zip de obs-studio en GitHub)
rem ===========================================================================

for %%i in ("%~dp0..") do set "PROJECT_ROOT=%%~fi"
set "OBS_DIR=%~1"
if "%OBS_DIR%"=="" set "OBS_DIR=C:\Program Files\obs-studio"
set "HDR_DIR=%~2"
if "%HDR_DIR%"=="" set "HDR_DIR=%PROJECT_ROOT%\.deps\include-32.2.2"

set "OBS_DLL=%OBS_DIR%\bin\64bit\obs.dll"
set "W32_DLL=%OBS_DIR%\bin\64bit\w32-pthreads.dll"

if not exist "%OBS_DLL%" (
    echo ERROR: no encuentro "%OBS_DLL%"
    echo        pasa la ruta de la instalacion de OBS como primer argumento.
    exit /b 1
)
if not exist "%HDR_DIR%\obs.h" (
    echo ERROR: no encuentro "%HDR_DIR%\obs.h"
    echo        pasa la carpeta de cabeceras de libobs como segundo argumento.
    exit /b 1
)

rem --- entorno de Visual Studio (dumpbin y lib) ----------------------------
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "!VSWHERE!" (
    for /f "usebackq delims=" %%p in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%p"
)
if defined VSROOT set "VCVARS=!VSROOT!\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS (
    echo ERROR: no encuentro vcvars64.bat; instala las Build Tools de Visual Studio
    exit /b 1
)
call "!VCVARS!" >nul 2>&1
if errorlevel 1 (
    echo ERROR: fallo al inicializar el entorno de Visual Studio
    exit /b 1
)

rem --- version del OBS instalado -------------------------------------------
set "OBS_FULL="
for /f "usebackq delims=" %%v in (`powershell -NoProfile -Command "(Get-Item '!OBS_DLL!').VersionInfo.ProductVersion"`) do set "OBS_FULL=%%v"
for /f "tokens=1,2,3 delims=." %%a in ("!OBS_FULL!") do set "OBS_VER=%%a.%%b.%%c"
if "!OBS_VER!"=="" (
    echo ERROR: no pude leer la version de "!OBS_DLL!" ^(valor: "!OBS_FULL!"^)
    exit /b 1
)
for /f "tokens=1 delims=." %%a in ("!OBS_VER!") do set "OBS_MAJOR=%%a"
echo OBS instalado: !OBS_FULL!

rem --- coherencia cabeceras contra version instalada ------------------------
set "HDR_VER="
for /f "tokens=3" %%a in ('findstr /R /C:"^#define LIBOBS_API_MAJOR_VER " "%HDR_DIR%\obs-config.h"') do set "HDR_MAJOR=%%a"
for /f "tokens=3" %%a in ('findstr /R /C:"^#define LIBOBS_API_MINOR_VER " "%HDR_DIR%\obs-config.h"') do set "HDR_MINOR=%%a"
for /f "tokens=3" %%a in ('findstr /R /C:"^#define LIBOBS_API_PATCH_VER " "%HDR_DIR%\obs-config.h"') do set "HDR_PATCH=%%a"
if defined HDR_MAJOR set "HDR_VER=!HDR_MAJOR!.!HDR_MINOR!.!HDR_PATCH!"
if not defined HDR_VER (
    echo AVISO: no pude leer LIBOBS_API_* de "%HDR_DIR%\obs-config.h"
) else if not "!HDR_VER!"=="!OBS_VER!" (
    echo AVISO: las cabeceras son de libobs !HDR_VER! y el OBS instalado es !OBS_VER!.
    echo        funciona mientras la API no cambie, pero lo suyo es compilar
    echo        contra las cabeceras de la misma version.
)

rem --- construir el SDK ------------------------------------------------------
set "SDK=%PROJECT_ROOT%\.deps\obs-sdk"
if exist "%SDK%" (
    echo Regenerando "%SDK%"...
    rmdir /s /q "%SDK%"
    if exist "%SDK%" (
        echo ERROR: no pude borrar "%SDK%"
        exit /b 1
    )
)
mkdir "%SDK%\include" "%SDK%\lib" "%SDK%\bin\64bit" "%SDK%\cmake" 2>nul

echo Copiando cabeceras desde "%HDR_DIR%"...
xcopy /E /I /Y /Q "%HDR_DIR%\*" "%SDK%\include\" >nul
if errorlevel 1 (
    echo ERROR: fallo xcopy de las cabeceras
    exit /b 1
)
rem w32-pthreads deja pthread.h y sched.h sueltos; libobs no los incluye,
rem pero el target importado OBS::w32-pthreads declara ese include dir.
if exist "%PROJECT_ROOT%\.deps\include\pthread.h" copy /y "%PROJECT_ROOT%\.deps\include\pthread.h" "%SDK%\include\" >nul
if exist "%PROJECT_ROOT%\.deps\include\sched.h" copy /y "%PROJECT_ROOT%\.deps\include\sched.h" "%SDK%\include\" >nul

echo Generando import lib de obs.dll...
call :make_implib "%OBS_DLL%" "%SDK%\lib\obs.lib" || exit /b 1
echo Generando import lib de w32-pthreads.dll...
call :make_implib "%W32_DLL%" "%SDK%\lib\w32-pthreads.lib" || exit /b 1

copy /y "%OBS_DLL%" "%SDK%\bin\64bit\" >nul
if errorlevel 1 (
    echo ERROR: no pude copiar obs.dll al SDK
    exit /b 1
)
if exist "%W32_DLL%" copy /y "%W32_DLL%" "%SDK%\bin\64bit\" >nul

copy /y "%~dp0obs-sdk\libobsConfig.cmake" "%SDK%\cmake\" >nul
if errorlevel 1 (
    echo ERROR: falta scripts\obs-sdk\libobsConfig.cmake
    exit /b 1
)
powershell -NoProfile -Command "(Get-Content '%~dp0obs-sdk\libobsConfigVersion.cmake.in' -Raw) -replace '@OBS_VERSION_MAJOR@', '!OBS_MAJOR!' -replace '@OBS_VERSION@', '!OBS_VER!' | Set-Content -Encoding ascii '%SDK%\cmake\libobsConfigVersion.cmake'"
if errorlevel 1 (
    echo ERROR: no pude generar libobsConfigVersion.cmake
    exit /b 1
)
echo !OBS_VER!>"%SDK%\version.txt"

echo.
echo SDK listo: "%SDK%"  ^(libobs !OBS_VER!^)
echo   CMakePresets.json ya apunta OBS_SDK_DIR ahi: el proximo configure
echo   del preset windows-x64 compila y enlaza contra este SDK.
echo   Para volver a libobs compilado desde fuente basta borrar el directorio.
exit /b 0

rem ---------------------------------------------------------------------------
rem  :make_implib <dll> <out.lib>
rem
rem  Genera una import lib desde los exports de un DLL: dumpbin para listar,
rem  un .def intermedio y lib.exe, el mismo metodo que usa la herramienta de
rem  OBS (genobslib.bat) pero filtrando la salida de dumpbin con cuidado: la
rem  tabla de exports tiene el formato "ordinal hint RVA nombre" y el resto
rem  del listado (cabecera, resumen) no debe acabar en el .def.
rem ---------------------------------------------------------------------------
:make_implib
set "DLL=%~1"
set "OUTLIB=%~2"
set "TMPD=%TEMP%\vtes_obs_sdk_%~n1"
if exist "%TMPD%" rmdir /s /q "%TMPD%"
mkdir "%TMPD%"

dumpbin /nologo /EXPORTS "%DLL%" > "%TMPD%\exports.txt" 2>&1
if errorlevel 1 (
    echo ERROR: dumpbin fallo sobre "%DLL%"
    type "%TMPD%\exports.txt"
    exit /b 1
)

rem La columna 4 de las lineas de tabla es el nombre. Entre columnas hace
rem falta al menos un espacio de verdad ( " *" es cero-o-mas espacios y "  *"
rem uno-o-mas): sin esa condicion, "3C81FF3E time date stamp" se trocea como
rem hex+hex+letra y pasa el filtro. La validacion como identificador C va en
rem una segunda pasada sobre el fichero: un pipe por iteracion (echo | findstr
rem dentro del for) falla en silencio y devuelve una lista vacia.
(
    for /f "tokens=4" %%n in ('findstr /R /C:"^ *[0-9][0-9]*  *[0-9A-Fa-f][0-9A-Fa-f]*  *[0-9A-Fa-f][0-9A-Fa-f]*  *[a-zA-Z_]" "%TMPD%\exports.txt"') do @echo %%n
) > "%TMPD%\names_raw.txt"
findstr /R /C:"^[a-zA-Z_][a-zA-Z0-9_]*$" "%TMPD%\names_raw.txt" > "%TMPD%\names.txt"

for %%s in ("%TMPD%\names.txt") do if %%~zs LSS 100 (
    echo ERROR: la lista de exports de "%DLL%" salio casi vacia
    exit /b 1
)

(
    echo LIBRARY %~n1.dll
    echo EXPORTS
    type "%TMPD%\names.txt"
) > "%TMPD%\exports.def"

lib /nologo /DEF:"%TMPD%\exports.def" /MACHINE:X64 /OUT:"%OUTLIB%" > "%TMPD%\lib.log" 2>&1
if errorlevel 1 (
    echo ERROR: lib.exe fallo generando "%OUTLIB%"
    type "%TMPD%\lib.log"
    exit /b 1
)
for %%s in ("%OUTLIB%") do echo   %%~nxs: %%~zs bytes

rmdir /s /q "%TMPD%"
exit /b 0
