#!/usr/bin/env python3
"""Genera cudart64_12.lib a partir de la cudart64_12.dll que despliega OBS.

Cuando el CUDA Toolkit no esta instalado pero TensorRT si (TENSORRT_ROOT
definido), CMake compila src/detection/tensorrt_detector.cpp, que llama a la
API de CUDA runtime, y el enlace falla con los simbolos cuda* sin resolver:
no hay cudart.lib que enlazar. La DLL que OBS ya tiene en obs-plugins/64bit
si sirve: se extraen sus exports con dumpbin y se construye una libreria de
importacion con lib.

El .lib resultante va a cudalib/, que es donde el CMakeLists lo busca cuando
CUDA_PATH no esta definido. Ejecutar con el Python de Windows:

    python scripts\\gen_cudart_lib.py

Comprobado: los 15 metodos cuda* que usa tensorrt_detector.cpp estan entre
los 442 simbolos que exporta la DLL.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'cudalib')
DLL = r'C:\Program Files\obs-studio\obs-plugins\64bit\cudart64_12.dll'

# Cadena de herramientas MSVC. Si falla, pásala por argumento:
#     python scripts\gen_cudart_lib.py "C:\...\bin\Hostx64\x64"
VS = (sys.argv[1] if len(sys.argv) > 1 else
      r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools'
      r'\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64')


def run(exe, args):
    r = subprocess.run([os.path.join(VS, exe)] + args,
                       capture_output=True, text=True, errors='replace')
    return r.stdout, r.stderr


def main():
    if not os.path.exists(DLL):
        print('ERROR: no existe %s' % DLL)
        print('Ajusta DLL al destino de despliegue de tu OBS.')
        return 1
    if not os.path.exists(os.path.join(VS, 'dumpbin.exe')):
        print('ERROR: no se encuentra dumpbin.exe en %s' % VS)
        return 1

    out, err = run('dumpbin.exe', ['/EXPORTS', DLL])
    syms = []
    for line in out.splitlines():
        m = re.match(r'\s*\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)\s*$', line)
        if m and m.group(1) != 'Summary':
            syms.append(m.group(1))
    if not syms:
        print('ERROR: no se han extraido simbolos de la DLL.')
        print((out + err)[:600])
        return 1
    print('simbolos exportados: %d' % len(syms))

    os.makedirs(OUTDIR, exist_ok=True)
    defpath = os.path.join(OUTDIR, 'cudart64_12.def')
    with open(defpath, 'w') as f:
        f.write('EXPORTS\n')
        for s in syms:
            f.write(s + '\n')

    libpath = os.path.join(OUTDIR, 'cudart64_12.lib')
    out2, err2 = run('lib.exe',
                     ['/DEF:' + defpath, '/OUT:' + libpath, '/MACHINE:X64'])
    if not os.path.exists(libpath):
        print('ERROR: lib.exe no ha generado %s' % libpath)
        print((out2 + err2)[:600])
        return 1
    print('generado: %s (%d bytes)' % (libpath, os.path.getsize(libpath)))
    print('El CMakeLists ya lo enlaza: reconfigura y compila.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
