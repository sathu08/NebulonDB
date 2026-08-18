@echo off
setlocal EnableExtensions

rem ============================================================
rem NebulonDB Installation Script (Windows)
rem Installs NebulonDB into a fixed location (%USERPROFILE%\.nebulondb)
rem and creates a global 'nebulondb' launcher in %USERPROFILE%\.local\bin.
rem ============================================================

set "REPO_URL=https://github.com/sathu08/NebulonDB.git"
set "BRANCH=master"

rem ------------------------------------------------------------
rem Check Git
rem ------------------------------------------------------------

where git >nul 2>nul
if errorlevel 1 (
    echo [NebulonDB][ERROR] Git is not installed. Please install Git first.
    exit /b 1
)

rem ------------------------------------------------------------
rem Clone or Update NebulonDB
rem ------------------------------------------------------------

echo [NebulonDB] NebulonDB repository:
echo [NebulonDB] %REPO_URL%
echo [NebulonDB] Target branch:
echo [NebulonDB] %BRANCH%

set "PROJECT_DIR=%USERPROFILE%\.nebulondb"

echo [NebulonDB] Install directory:
echo [NebulonDB] %PROJECT_DIR%

if exist "%PROJECT_DIR%\.git" goto :update_repo

rem ---------- Fresh clone: create a NebulonDB folder and clone into it ----------
if exist "%PROJECT_DIR%" (
    echo [NebulonDB][ERROR] NebulonDB directory already exists but is not a Git repository: %PROJECT_DIR%
    exit /b 1
)

echo [NebulonDB] Cloning NebulonDB into:
echo [NebulonDB] %PROJECT_DIR%
echo [NebulonDB] Branch: %BRANCH%

mkdir "%PROJECT_DIR%"

git clone --branch %BRANCH% --single-branch %REPO_URL% "%PROJECT_DIR%"
if errorlevel 1 goto :error

echo [NebulonDB] Repository cloned successfully.
goto :movedir

:update_repo
echo [NebulonDB] NebulonDB repository already exists:
echo [NebulonDB] %PROJECT_DIR%

echo [NebulonDB] Fetching latest changes from branch '%BRANCH%'...
git fetch origin %BRANCH%
if errorlevel 1 goto :error

echo [NebulonDB] Switching to branch '%BRANCH%'...
git show-ref --verify --quiet refs/heads/%BRANCH%
if errorlevel 1 (
    git checkout -b %BRANCH% --track origin/%BRANCH%
) else (
    git checkout %BRANCH%
)
if errorlevel 1 goto :error

echo [NebulonDB] Updating branch '%BRANCH%'...
git pull --ff-only origin %BRANCH%
if errorlevel 1 goto :error

:movedir
rem ------------------------------------------------------------
rem Set Current Directory to NebulonDB
rem ------------------------------------------------------------

cd /d "%PROJECT_DIR%"
if errorlevel 1 goto :error

echo [NebulonDB] NebulonDB Home:
echo [NebulonDB] %PROJECT_DIR%

rem ------------------------------------------------------------
rem Verify Branch
rem ------------------------------------------------------------

for /f "delims=" %%i in ('git branch --show-current') do set "CURRENT_BRANCH=%%i"

if not "%CURRENT_BRANCH%"=="%BRANCH%" (
    echo [NebulonDB][ERROR] Expected branch '%BRANCH%', but currently on '%CURRENT_BRANCH%'.
    exit /b 1
)

echo [NebulonDB] Git branch:
echo [NebulonDB] %CURRENT_BRANCH%

rem ------------------------------------------------------------
rem Check / Install uv
rem ------------------------------------------------------------

where uv >nul 2>nul
if errorlevel 1 (
    echo [NebulonDB] uv is not installed.
    echo [NebulonDB] Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :error
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [NebulonDB][ERROR] uv installation failed or uv is not available in PATH.
    exit /b 1
)

echo [NebulonDB] uv version:
uv --version

rem ------------------------------------------------------------
rem Check / Install Python 3.10
rem ------------------------------------------------------------

set "PYTHON_VERSION=3.10"

echo [NebulonDB] Checking Python %PYTHON_VERSION%...

uv python find %PYTHON_VERSION% >nul 2>nul
if errorlevel 1 (
    echo [NebulonDB] Python %PYTHON_VERSION% not found.
    echo [NebulonDB] Installing Python %PYTHON_VERSION%...
    uv python install %PYTHON_VERSION%
    if errorlevel 1 goto :error
)

for /f "delims=" %%i in ('uv python find %PYTHON_VERSION%') do set "PYTHON_PATH=%%i"

echo [NebulonDB] Using Python:
echo [NebulonDB] %PYTHON_PATH%

rem ------------------------------------------------------------
rem Install NebulonDB
rem ------------------------------------------------------------

echo [NebulonDB] Installing NebulonDB dependencies...

uv sync --python %PYTHON_VERSION%
if errorlevel 1 goto :error

rem ------------------------------------------------------------
rem Optional ML / Vector Extras (off by default)
rem
rem NEBULONDB_INSTALL_ML controls the torch build:
rem   1     -> CPU-only torch (small, works everywhere)
rem   1_CPU -> same as 1 (explicit CPU)
rem   CPU   -> same as 1
rem   GPU   -> CUDA (cu124) torch for NVIDIA GPUs
rem ------------------------------------------------------------

set "ML_MODE=0"
if defined NEBULONDB_INSTALL_ML for /f "tokens=*" %%i in ('echo %NEBULONDB_INSTALL_ML%') do set "ML_MODE=%%i"

if /I "%ML_MODE%"=="1" goto ml_cpu
if /I "%ML_MODE%"=="1_cpu" goto ml_cpu
if /I "%ML_MODE%"=="cpu" goto ml_cpu
if /I "%ML_MODE%"=="gpu" goto ml_gpu
if /I "%ML_MODE%"=="1_gpu" goto ml_gpu
if /I "%ML_MODE%"=="0" goto ml_done
echo [NebulonDB][ERROR] NEBULONDB_INSTALL_ML must be 1, 1_CPU, CPU, GPU or unset.
exit /b 1

:ml_cpu
echo [NebulonDB] Installing optional ML/vector extras (CPU-only torch)...
uv sync --python %PYTHON_VERSION% --extra ml-cpu
if errorlevel 1 goto :error
goto ml_done

:ml_gpu
echo [NebulonDB] Installing optional ML/vector extras (CUDA cu124 torch for GPU)...
uv sync --python %PYTHON_VERSION% --extra ml-gpu
if errorlevel 1 goto :error
echo [NebulonDB] Swapping torch to the CUDA cu124 build...
uv pip install --python %PYTHON_VERSION% torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :error
goto ml_done

:ml_done

rem ------------------------------------------------------------
rem Install Global NebulonDB CLI
rem ------------------------------------------------------------

set "VENV_DIR=%PROJECT_DIR%\.venv"
set "CLI_PATH=%VENV_DIR%\Scripts\nebulondb.exe"
set "BIN_DIR=%USERPROFILE%\.local\bin"
set "GLOBAL_CLI=%BIN_DIR%\nebulondb.cmd"

if not exist "%CLI_PATH%" (
    echo [NebulonDB][ERROR] NebulonDB CLI was not created: %CLI_PATH%
    exit /b 1
)

echo [NebulonDB] Installing global NebulonDB CLI...

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

> "%GLOBAL_CLI%" (
    echo @echo off
    echo set "NEBULONDB_HOME=%PROJECT_DIR%"
    echo "%CLI_PATH%" %%*
)

rem ------------------------------------------------------------
rem Configure ~/.local/bin in PATH
rem ------------------------------------------------------------

echo %PATH% | findstr /i /c:"%BIN_DIR%" >nul 2>nul
if errorlevel 1 (
    setx PATH "%BIN_DIR%;%PATH%" >nul
    if errorlevel 1 goto :error
)

set "PATH=%BIN_DIR%;%PATH%"

rem ------------------------------------------------------------
rem Verify NebulonDB CLI
rem ------------------------------------------------------------

where nebulondb >nul 2>nul
if errorlevel 1 (
    echo [NebulonDB][ERROR] NebulonDB CLI was not installed correctly.
    exit /b 1
)

echo [NebulonDB] NebulonDB executable:
where nebulondb

echo [NebulonDB] Testing NebulonDB CLI...
nebulondb --help
if errorlevel 1 goto :error

rem ------------------------------------------------------------
rem Installation Complete
rem ------------------------------------------------------------

echo.
echo ============================================================
echo  NebulonDB Installation Complete
echo ============================================================
echo.
echo Repository     : %REPO_URL%
echo Branch         : %CURRENT_BRANCH%
echo Directory      : %PROJECT_DIR%
echo Python         : %PYTHON_PATH%
echo Virtual Env    : %VENV_DIR%
echo NebulonDB CLI  : %BIN_DIR%\nebulondb.cmd
echo NEBULONDB_HOME : %PROJECT_DIR%
echo.
echo Open a new terminal, then start NebulonDB from anywhere:
echo.
echo     nebulondb start
echo.
echo ============================================================
exit /b 0

:error
echo [NebulonDB][ERROR] Installation failed.
exit /b 1
