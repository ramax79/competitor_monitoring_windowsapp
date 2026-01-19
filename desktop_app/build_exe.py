"""
Скрипт для сборки .exe файла из PyQt6 приложения
Использование: python build_exe.py
"""
import PyInstaller.__main__
import sys
from pathlib import Path

# Путь к директории скрипта
script_dir = Path(__file__).parent

PyInstaller.__main__.run([
    'main.py',
    '--name=CompetitorMonitor',
    '--onefile',
    '--windowed',
    '--icon=NONE',  # Можно добавить иконку: --icon=icon.ico
    '--add-data=README.md;.',
    '--hidden-import=PyQt6.QtCore',
    '--hidden-import=PyQt6.QtGui',
    '--hidden-import=PyQt6.QtWidgets',
    '--collect-all=PyQt6',
    '--clean',
    '--noconfirm',
])

print("\n✓ Сборка завершена!")
print(f"✓ .exe файл находится в: {script_dir / 'dist' / 'CompetitorMonitor.exe'}")
