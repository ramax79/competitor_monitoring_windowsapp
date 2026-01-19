# Инструкция по установке и сборке

## Быстрый старт

### 1. Установка зависимостей

```bash
cd desktop_app
pip install -r requirements.txt
```

### 2. Запуск backend

В корне проекта (не в desktop_app):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Запуск приложения

```bash
python main.py
```

## Сборка в .exe

### Способ 1: Автоматическая сборка (Windows)

Просто запустите:

```bash
build.bat
```

Скрипт автоматически:
- Проверит наличие Python
- Установит зависимости
- Соберёт .exe файл

Результат: `dist/CompetitorMonitor.exe`

### Способ 2: Через PyInstaller напрямую

```bash
pyinstaller CompetitorMonitor.spec --clean --noconfirm
```

### Способ 3: Через build_exe.py

```bash
python build_exe.py
```

## Размер .exe файла

Ожидаемый размер: **50-100 МБ**

Это нормально, так как PyInstaller включает:
- Python интерпретатор
- PyQt6 библиотеки
- Все зависимости

## Распространение .exe

После сборки файл `CompetitorMonitor.exe` можно:
- Копировать на другие компьютеры
- Распространять без установки Python
- Запускать напрямую (но требуется запущенный backend)

## Устранение проблем

### Ошибка: "Backend недоступен"

1. Убедитесь, что backend запущен:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Проверьте, что порт 8000 свободен

3. Если backend на другом адресе, измените `API_BASE` в `main.py`

### Ошибка при сборке: "PyInstaller не найден"

```bash
pip install pyinstaller
```

### Большой размер .exe

Это нормально. PyQt6 — большая библиотека. Можно попробовать:
- Использовать `--exclude-module` для исключения неиспользуемых модулей
- Использовать UPX для сжатия (уже включено в spec)

### Приложение не запускается

1. Проверьте, что все зависимости установлены
2. Запустите через консоль для просмотра ошибок:
   ```bash
   python main.py
   ```

## Настройка иконки

1. Создайте файл `icon.ico` (можно использовать онлайн-конвертеры)
2. В `CompetitorMonitor.spec` измените:
   ```python
   icon='icon.ico',
   ```
3. Пересоберите приложение
