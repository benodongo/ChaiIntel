@echo off
REM ============================================================
REM  ChaiIntel - One-click launcher (Windows)
REM  Creates the virtual environment if missing, installs
REM  dependencies, applies migrations, then starts the server.
REM ============================================================

setlocal
cd /d "%~dp0"

REM 1. Create venv if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the virtual environment.
        echo Make sure Python 3.9+ is installed and on your PATH.
        pause
        exit /b 1
    )
)

REM 2. Activate venv
call "venv\Scripts\activate.bat"

REM 3. Install / update dependencies only when requirements.txt has changed.
REM    We store the hash of requirements.txt in venv\.requirements.sha256 and
REM    compare it to the current file's hash on every launch.
set "REQ_HASH_FILE=venv\.requirements.sha256"
for /f "skip=1 tokens=* delims=" %%H in ('certutil -hashfile requirements.txt SHA256 ^| findstr /v ":"') do (
    set "CURRENT_HASH=%%H"
    goto :got_hash
)
:got_hash
set "CURRENT_HASH=%CURRENT_HASH: =%"

set "SAVED_HASH="
if exist "%REQ_HASH_FILE%" set /p SAVED_HASH=<"%REQ_HASH_FILE%"

if /i "%CURRENT_HASH%"=="%SAVED_HASH%" (
    echo Dependencies already up to date - skipping install.
) else (
    echo Installing dependencies (this may take a few minutes the first time)...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
    > "%REQ_HASH_FILE%" echo %CURRENT_HASH%
)

REM 4. Apply database migrations
python manage.py migrate

REM 5. Run the development server
echo.
echo ============================================================
echo  ChaiIntel is starting at http://127.0.0.1:8000
echo  Press CTRL+C in this window to stop the server.
echo ============================================================
echo.
python manage.py runserver

pause
endlocal
