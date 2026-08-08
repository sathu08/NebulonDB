@echo off
setlocal EnableExtensions

rem ============================================================
rem NebulonDB Installation Script (Windows)
rem ============================================================

set "REPO_URL=https://github.com/sathu08/NebulonDB.git"
set "BRANCH=dev"
set "PROJECT_DIR_NAME=NebulonDB"

set "INSTALL_BASE_DIR=%USERPROFILE%\CodeBase"
set "PROJECT_DIR=%INSTALL_BASE_DIR%\%PROJECT_DIR_NAME%"

rem ------------------------------------------------------------
rem Check Git
rem ------------------------------------------------------------

where git >nul 2>nul
if errorlevel 1 (
    echo [NebulonDB][ERROR] Git is not installed. Please install Git first.
    exit /b 1
)

rem ------------------------------------------------------------
rem Clone NebulonDB
rem ------------------------------------------------------------

echo [NebulonDB] NebulonDB repository:
echo [NebulonDB] %REPO_URL%
echo [NebulonDB] Target branch:
echo [NebulonDB] %BRANCH%

if not exist "%INSTALL_BASE_DIR%" mkdir "%INSTALL_BASE_DIR%"

if exist "%PROJECT_DIR%\.git" (

    echo [NebulonDB] NebulonDB repository already exists:
    echo [NebulonDB] %PROJECT_DIR%

    pushd "%PROJECT_DIR%"

    echo [NebulonDB] Fetching latest changes from branch '%BRANCH%'...
    git fetch origin %BRANCH%
    if errorlevel 1 goto :error

    echo [NebulonDB] Switching to branch '%BRANCH%'...
    git checkout %BRANCH%
    if errorlevel 1 goto :error

    echo [NebulonDB] Updating branch '%BRANCH%'...
    git pull --ff-only origin %BRANCH%
    if errorlevel 1 goto :error

    popd

) else (

    if exist "%PROJECT_DIR%" (
        echo [NebulonDB][ERROR] Directory already exists but is not a Git repository: %PROJECT_DIR%
        exit /b 1
    )

    echo [NebulonDB] Cloning NebulonDB...
    echo [NebulonDB] Branch: %BRANCH%

    git clone --branch %BRANCH% --single-branch %REPO_URL% "%PROJECT_DIR%"
    if errorlevel 1 goto :error

    echo [NebulonDB] Repository cloned successfully.
)

rem ------------------------------------------------------------
rem Move to Project Directory
rem ------------------------------------------------------------

pushd "%PROJECT_DIR%"
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
rem Verify NebulonDB CLI
rem ------------------------------------------------------------

set "VENV_DIR=%PROJECT_DIR%\.venv"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [NebulonDB][ERROR] Virtual environment was not created: %VENV_DIR%
    exit /b 1
)

echo [NebulonDB] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

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
rem Configure NEBULONDB_HOME
rem ------------------------------------------------------------

echo [NebulonDB] Configuring NEBULONDB_HOME...
setx NEBULONDB_HOME "%PROJECT_DIR%" >nul
if errorlevel 1 goto :error

rem ------------------------------------------------------------
rem Installation Complete
rem ------------------------------------------------------------

popd

echo.
echo ============================================================
echo  NebulonDB Installation Complete
echo ============================================================
echo.
echo Repository     : %REPO_URL%
echo Branch         : %CURRENT_BRANCH%
echo Project        : %PROJECT_DIR%
echo Python         : %PYTHON_PATH%
echo Virtual Env    : %VENV_DIR%
echo NebulonDB CLI  : %PROJECT_DIR%\.venv\Scripts\nebulondb.exe
echo NEBULONDB_HOME : %PROJECT_DIR%
echo.
echo Open a new terminal, then run:
echo.
echo     nebulondb start
echo.
echo ============================================================
exit /b 0

:error
echo [NebulonDB][ERROR] Installation failed.
exit /b 1
