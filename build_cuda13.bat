@echo off
setlocal

REM CUDA 13.4 (instalado) y la variante de TensorRT que le corresponde.
REM TENSORRT_ROOT en el entorno de Windows apunta a la 12.x, que no vale.
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4
set TENSORRT_ROOT=C:\Program Files\TensorRT-11.1.0.106-13.x

echo CUDA_PATH      = %CUDA_PATH%
echo TENSORRT_ROOT  = %TENSORRT_ROOT%
echo.

cd /d D:\Hermes\vtes

echo === Limpiando cache stale de CUDA 12.4 ===
REM find_library cachea su resultado: las entradas de cuando el toolkit era
REM v12.4 siguen ahi y apuntan a un cudart.lib que ya no existe.
set CMAKE=C:\Program Files\CMake\bin\cmake.exe
"%CMAKE%" -B build_x64 -U CUDART_LIB -U CUDA_CUDART -U CUDA_DRIVER_LIB -U CUDA_OpenCL_LIBRARY -U CUDA_bin2c_EXECUTABLE -U CUDA_cuFile_LIBRARY -U CUDA_cuFile_rdma_LIBRARY -U CUDA_cuFile_rdma_static_LIBRARY -U CUDA_cuFile_static_LIBRARY -U CUDA_cudart_LIBRARY -U CUDA_cudart_static_LIBRARY -U TENSORRT_INCLUDE_DIR -U TENSORRT_LIB -U TENSORRT_PLUGIN_LIB > nul 2>&1

echo === Configurando ===
"%CMAKE%" -S . -B build_x64 -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=RelWithDebInfo > build_x64\cfg.log 2>&1
if errorlevel 1 (
  echo ERROR en configure
  findstr /I "error" build_x64\cfg.log
  exit /b 1
)

echo === Compilando (RelWithDebInfo) ===
"%CMAKE%" --build build_x64 --config RelWithDebInfo --target vtes-card-scanner > build_x64\bld.log 2>&1

echo === Instalando a prefix de prueba ===
REM --config RelWithDebInfo es OBLIGATORIO. Sin el, el script de install de
REM OpenCV busca los .lib en lib/Release/ (config por defecto) y falla antes
REM de llegar a nuestras reglas, asi que no se despliega el plugin.
set PREFIX=%TEMP%\vtes_install
if exist "%PREFIX%" rmdir /s /q "%PREFIX%"
"%CMAKE%" --install build_x64 --prefix "%PREFIX%" --config RelWithDebInfo > build_x64\inst.log 2>&1
if errorlevel 1 (
  echo ERROR en install
  findstr /C:"CMake Error" build_x64\inst.log
  exit /b 1
)
echo|set /p=Ficheros instalados:
findstr /C:"-- Installing:" build_x64\inst.log | find /C /V ""

echo.
echo === DLLs del plugin ===
dir /b "%PREFIX%\obs-plugins\64bit\"

echo|set /p=Errores C:
findstr /C:"error C" build_x64\bld.log | find /C /V ""
echo|set /p=Errores LNK:
findstr /C:"error LNK" build_x64\bld.log | find /C /V ""
echo.
if exist build_x64\RelWithDebInfo\vtes-card-scanner.dll (
  echo DLL generada: build_x64\RelWithDebInfo\vtes-card-scanner.dll
) else (
  echo NO se genero la DLL. Primer error:
  findstr /C:"error " build_x64\bld.log | more +0
)
