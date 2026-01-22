@echo off
setlocal EnableDelayedExpansion

REM Check for running process
tasklist /FI "IMAGENAME eq slime_world.exe" 2>nul | find /I "slime_world.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo WARNING: slime_world.exe is currently running.
    set /p "CONFIRM=Kill it and continue? [Y/N]: "
    if /i "!CONFIRM!" NEQ "Y" (
        echo Build cancelled.
        exit /b 1
    )
    taskkill /F /IM slime_world.exe >nul 2>&1
)

if not exist build mkdir build

REM Find CUDA installation
set "CUDA_PATH="
for /f "delims=" %%i in ('dir /b /ad /o-n "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do (
    if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%i"
)
if not defined CUDA_PATH (
    echo ERROR: CUDA not found!
    exit /b 1
)
echo Found CUDA: %CUDA_PATH%

REM Find Visual Studio installation
set "VSDIR="
for /f "delims=" %%i in ('dir /b /ad /o-n "C:\Program Files\Microsoft Visual Studio\*" 2^>nul') do (
    if not defined VSDIR (
        for /f "delims=" %%j in ('dir /b /ad "C:\Program Files\Microsoft Visual Studio\%%i" 2^>nul ^| findstr /i "Enterprise Professional Community BuildTools"') do (
            if not defined VSDIR set "VSDIR=C:\Program Files\Microsoft Visual Studio\%%i\%%j"
        )
    )
)
if not defined VSDIR (
    echo ERROR: Visual Studio not found!
    exit /b 1
)
echo Found VS: %VSDIR%

REM Find MSVC tools version
set "VCTOOLS="
for /f "delims=" %%i in ('dir /b /ad /o-n "%VSDIR%\VC\Tools\MSVC" 2^>nul') do (
    if not defined VCTOOLS set "VCTOOLS=%VSDIR%\VC\Tools\MSVC\%%i"
)
if not defined VCTOOLS (
    echo ERROR: MSVC tools not found!
    exit /b 1
)
echo Found MSVC: %VCTOOLS%

REM Find Windows SDK
set "WINSDK=C:\Program Files (x86)\Windows Kits\10"
set "WINSDKVER="
for /f "delims=" %%i in ('dir /b /ad /o-n "%WINSDK%\Include" 2^>nul') do (
    if not defined WINSDKVER set "WINSDKVER=%%i"
)
if not defined WINSDKVER (
    echo ERROR: Windows SDK not found!
    exit /b 1
)
echo Found Windows SDK: %WINSDKVER%

set "PATH=%CUDA_PATH%\bin;%VCTOOLS%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%VCTOOLS%\include;%WINSDK%\Include\%WINSDKVER%\ucrt;%WINSDK%\Include\%WINSDKVER%\shared;%WINSDK%\Include\%WINSDKVER%\um"
set "LIB=%VCTOOLS%\lib\x64;%WINSDK%\Lib\%WINSDKVER%\ucrt\x64;%WINSDK%\Lib\%WINSDKVER%\um\x64"

echo.
echo Building Slime World Pattern Finder (CUDA)...

cd build
nvcc -O3 -arch=sm_89 -std=c++17 -allow-unsupported-compiler -o ..\slime_world.exe ..\slime_world.cu

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
) else (
    echo Build failed!
)
