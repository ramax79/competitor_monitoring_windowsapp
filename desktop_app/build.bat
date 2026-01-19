@echo off
echo ====================================
echo Сборка CompetitorMonitor.exe
echo ====================================
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    pause
    exit /b 1
)

REM Проверка наличия PyInstaller
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Установка PyInstaller...
    pip install pyinstaller
)

echo Установка зависимостей...
pip install -r requirements.txt

echo.
echo Сборка .exe файла...
echo.

REM Сборка через spec файл
pyinstaller CompetitorMonitor.spec --clean --noconfirm

if exist "dist\CompetitorMonitor.exe" (
    echo.
    echo ====================================
    echo СБОРКА УСПЕШНА!
    echo ====================================
    echo.
    echo Файл: dist\CompetitorMonitor.exe
    echo Размер: 
    dir "dist\CompetitorMonitor.exe" | find "CompetitorMonitor.exe"
    echo.
) else (
    echo.
    echo ====================================
    echo ОШИБКА СБОРКИ!
    echo ====================================
    echo.
)

pause
