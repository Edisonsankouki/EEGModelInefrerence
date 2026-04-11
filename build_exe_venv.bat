@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  exit /b 1
)

echo [1/3] Using venv Python:
".venv\Scripts\python.exe" -V
if errorlevel 1 exit /b 1

echo [2/3] Cleaning old build output...
if exist "build\exe" rmdir /s /q "build\exe"
if exist "dist\gpt2ref" rmdir /s /q "dist\gpt2ref"

echo [3/3] Building with exe.spec...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean exe.spec
if errorlevel 1 (
  echo [ERROR] PyInstaller build failed.
  exit /b 1
)

echo.
echo Build success: dist\gpt2ref\gpt2ref.exe
echo Test command:
echo dist\gpt2ref\gpt2ref.exe Report_generation 20370.edf

endlocal
