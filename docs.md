## Структура проекта

- `backend/`
  - `main.py` — приложение FastAPI, все endpoint’ы:
    - `/analyze_text`
    - `/analyze_image`
    - `/parse_demo`
    - `/history`
    - `/` (health‑check)
  - `openai_client.py` — обёртки вокруг OpenAI API:
    - `analyze_competitor_text(text)`
    - `analyze_competitor_image(image_bytes, mime_type)`
  - `history.py` — простое файловое хранилище истории в `history.json`.
  - `schemas.py` — Pydantic‑схемы запросов/ответов.

- `frontend/`
  - `index.html` — минимальное веб‑приложение (HTML + JS).

- `requirements.txt` — зависимости Python.
- `history.json` —_файл истории (автоматически создаётся при первом запросе).
- `README.md` — описание проекта и инструкция по запуску.
- `docs.md` — эта документация.

---

## API

### `POST /analyze_text`

- **Назначение**: Анализ конкурентного текста.
- **Тело запроса (JSON)**:

```json
{
  "text": "Полотно текста конкурента..."
}
```

- **Успешный ответ (200, JSON)**:

```json
{
  "strengths": ["..."],
  "weaknesses": ["..."],
  "unique_offers": ["..."],
  "recommendations": ["..."]
}
```

- **Ошибки**:
  - `400` — если `text` пустой.
  - `500` — ошибка взаимодействия с моделью.

#### Пример запроса (curl)

```bash
curl -X POST http://localhost:8000/analyze_text ^
  -H "Content-Type: application/json" ^
  -d "{ \"text\": \"Наш продукт — лучший в мире...\" }"
```

---

### `POST /analyze_image`

- **Назначение**: Мультимодальный анализ изображения конкурента.
- **Тело запроса**: `multipart/form-data` с полем:
  - `file` — файл изображения (тип `image/*`).

- **Успешный ответ (200, JSON)**:

```json
{
  "description": "Что изображено на баннере/сайте/упаковке",
  "marketing_insights": ["..."],
  "visual_style_evaluation": ["..."]
}
```

- **Ошибки**:
  - `400` — если не передан файл, файл не является изображением или пуст.
  - `500` — ошибка взаимодействия с моделью.

#### Пример запроса (curl, PowerShell)

```bash
curl -X POST http://localhost:8000/analyze_image ^
  -F "file=@banner.png"
```

---

### `POST /parse_demo`

- **Назначение**: Демо‑парсинг лендинга конкурента с последующим анализом.
- **Тело запроса (JSON)**:

```json
{
  "url": "https://пример-конкурента.ru"
}
```

- **Логика**:
  1. Выполнить HTTP‑запрос к указанному URL.
  2. Извлечь:
     - `<title>`
     - первый `<h1>`
     - первый `<p>`
  3. Сформировать текст для анализа и отправить его в OpenAI.

- **Успешный ответ (200, JSON)**:

```json
{
  "url": "https://пример-конкурента.ru",
  "title": "Заголовок страницы или null",
  "h1": "Первый H1 или null",
  "first_paragraph": "Первый абзац или null",
  "analysis": {
    "strengths": ["..."],
    "weaknesses": ["..."],
    "unique_offers": ["..."],
    "recommendations": ["..."]
  }
}
```

- **Ошибки**:
  - `400` — если URL недостижим или ответ с ошибкой.
  - `500` — ошибка взаимодействия с моделью.

#### Пример запроса (curl)

```bash
curl -X POST http://localhost:8000/parse_demo ^
  -H "Content-Type: application/json" ^
  -d "{ \"url\": \"https://example.com\" }"
```

---

### `GET /history`

- **Назначение**: Вернуть последние 10 запросов (текст, изображение, parse_demo).
- **Успешный ответ (200, JSON)**:

```json
{
  "items": [
    {
      "timestamp": "2024-01-01T12:00:00.000000Z",
      "type": "text",
      "payload": {
        "input_preview": "Начало анализируемого текста...",
        "analysis": {
          "strengths": [],
          "weaknesses": [],
          "unique_offers": [],
          "recommendations": []
        }
      }
    }
  ]
}
```

#### Пример запроса (curl)

```bash
curl http://localhost:8000/history
```

---

### `GET /`

- **Назначение**: Простой health‑check.
- **Ответ (200, JSON)**:

```json
{
  "status": "ok",
  "message": "Competitor Monitoring Assistant backend is running."
}
```

---

## Мультимодальные функции

### Текстовая аналитика

- Используется OpenAI‑модель (по умолчанию `gpt-4o-mini`).
- Системный prompt просит модель вернуть **строго JSON** с ключами:
  - `strengths`
  - `weaknesses`
  - `unique_offers`
  - `recommendations`
- Ответ парсится как JSON; если модель вернула невалидный JSON — оборачивается в безопасный формат.

### Аналитика изображений

- Используется мультимодальный режим (`image_url` + `data:` URL).
- Модель возвращает JSON c ключами:
  - `description`
  - `marketing_insights`
  - `visual_style_evaluation`
- В UI блоки отображаются как человекочитаемые списки.

---

## История запросов

- Файл: `history.json` в корне проекта.
- Максимум **10** последних записей.
- Формат элемента:

```json
{
  "timestamp": "ISO8601 UTC",
  "type": "text | image | parse_demo",
  "payload": {
    "...": "минимально необходимые данные для UI"
  }
}
```

- Добавление и чтение истории происходит через модуль `backend/history.py` с файловой блокировкой, чтобы избежать гонок записи.

---

## Фронтенд (мини‑UI)

- Один файл `frontend/index.html`:
  - табы:
    - **Текст конкурента** (отправка в `/analyze_text`);
    - **Изображение** (загрузка файла → `/analyze_image`);
    - **URL конкурента** (демо‑парсинг через `/parse_demo`).
  - блок результатов:
    - четыре карточки: сильные, слабые стороны, УТП и рекомендации;
    - отдельные блоки для описания и визуального стиля (для изображений).
  - блок истории:
    - подгружается из `/history`;
    - показывает тип запроса и короткий превью‑текст / URL / имя файла.

UI подключается к backend по `http://localhost:8000` (см. константу `API_BASE` в JS‑скрипте).

