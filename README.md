# Competitor Monitoring Assistant (MVP)

Мультимодальное MVP‑приложение для мониторинга конкурентов: анализирует тексты, изображения и демо‑парсит лендинги конкурентов с помощью OpenAI (GPT‑4.1 / GPT‑4o‑mini).

Backend реализован на **FastAPI**, фронтенд — минимальная одностраничная HTML+JS‑страница.

---

## Возможности

- **Анализ конкурентного текста**
  - Принимает произвольный маркетинговый текст конкурента (лендинг, офферы, промо‑кампании).
  - Возвращает структурированную аналитику:
    - сильные стороны;
    - слабые стороны;
    - уникальные предложения;
    - рекомендации по улучшению стратегии.

- **Анализ изображений**
  - Принимает изображение (баннер, скриншот сайта, упаковку товара).
  - Использует мультимодальную модель OpenAI.
  - Возвращает:
    - описание изображения;
    - маркетинговые инсайты;
    - оценку визуального стиля конкурента.

- **Демо‑парсинг внешних источников**
  - Endpoint `/parse_demo` принимает URL.
  - Выполняет HTTP‑запрос, извлекает:
    - `<title>`;
    - первый `<h1>`;
    - первый абзац `<p>`.
  - Отправляет извлечённый текст в модель для конкурентного анализа.

- **История запросов**
  - Последние **10** запросов (текст, изображения, parse_demo) сохраняются в `history.json`.
  - Endpoint `/history` возвращает историю для отображения в UI.

---

## Стек

- Python 3.10+
- FastAPI
- Uvicorn
- OpenAI (GPT‑4.1 / GPT‑4o‑mini, Vision)
- Selenium, BeautifulSoup4
- Мини‑фронтенд на чистом HTML + JS
- **Desktop приложение на PyQt6** (в папке `desktop_app/`)

---

## Установка и запуск

1. **Клонировать / распаковать проект**

   ```bash
   cd competitor_monitoring_windowsapp
   ```

2. **Создать и активировать виртуальное окружение (рекомендуется)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows PowerShell
   ```

3. **Установить зависимости**

   ```bash
   pip install -r requirements.txt
   ```

4. **Создать файл `.env`**

   В корне проекта создайте файл `.env` со значениями:

   ```bash
   OPENAI_API_KEY=ВАШ_API_КЛЮЧ
   OPENAI_MODEL_TEXT=gpt-4o-mini        # (опционально, можно поменять)
   OPENAI_MODEL_IMAGE=gpt-4o-mini       # (опционально, можно поменять)
   ```

   Библиотека `python-dotenv` автоматически подгружает этот файл при старте backend.

5. **Запустить backend**

   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Открыть UI**

   - Открыть файл `frontend/index.html` в браузере
     (или поднять любой простой статический сервер).

   Приложение будет отправлять запросы на `http://localhost:8000`.

---

## Реализованные endpoint’ы (FastAPI)

- `POST /analyze_text`
  - Вход: JSON `{ "text": "..." }`
  - Выход: JSON
    ```json
    {
      "strengths": ["..."],
      "weaknesses": ["..."],
      "unique_offers": ["..."],
      "recommendations": ["..."]
    }
    ```

- `POST /analyze_image`
  - Вход: `multipart/form-data` с полем `file` (image/*).
  - Выход: JSON:
    ```json
    {
      "description": "...",
      "marketing_insights": ["..."],
      "visual_style_evaluation": ["..."]
    }
    ```

- `POST /parse_demo`
  - Вход: JSON `{ "url": "https://..." }`
  - Логика:
    - запрашивает страницу;
    - извлекает title + h1 + первый абзац;
    - передаёт текст в модель для анализа конкурента.
  - Выход: JSON:
    ```json
    {
      "url": "https://...",
      "title": "...",
      "h1": "...",
      "first_paragraph": "...",
      "analysis": {
        "strengths": [],
        "weaknesses": [],
        "unique_offers": [],
        "recommendations": []
      }
    }
    ```

- `GET /history`
  - Выход: JSON с последними 10 запросами (см. `docs.md`).

---

## Как это связано с заданием

- **Python backend (FastAPI)** — реализован в `backend/main.py`.
- **Мультимодальность (текст + изображения)** — функции `analyze_text` и `analyze_image`.
- **Сбор данных из внешних источников (демо)** — endpoint `/parse_demo`.
- **История** — хранится в `history.json`, логика в `backend/history.py`.
- **Мини‑UI** — одностраничник `frontend/index.html`:
  - поле ввода текста;
  - поле загрузки изображения;
  - поле для URL;
  - кнопка «Проанализировать»;
  - блок результатов;
  - блок истории запросов.

Дополнительные технические детали, структура проекта и примеры запросов описаны в `docs.md`.

---

## Desktop приложение (PyQt6)

В папке `desktop_app/` находится десктопное приложение на PyQt6, которое полностью повторяет функционал веб-версии.

### Быстрый запуск

1. **Установите зависимости:**
   ```bash
   cd desktop_app
   pip install -r requirements.txt
   ```

2. **Запустите backend** (в корне проекта):
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Запустите приложение:**
   ```bash
   python desktop_app/main.py
   ```

### Сборка в .exe

```bash
cd desktop_app
build.bat
```

Или вручную:
```bash
pyinstaller CompetitorMonitor.spec --clean --noconfirm
```

Готовый `.exe` файл будет в `desktop_app/dist/CompetitorMonitor.exe`.

Подробные инструкции: см. `desktop_app/README.md` и `desktop_app/INSTALL.md`.

