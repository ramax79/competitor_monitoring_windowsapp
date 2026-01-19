import logging
from typing import List, Optional

from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

from . import history as history_store
from . import openai_client
from .schemas import (
    AnalyzeImageResponse,
    AnalyzeTextRequest,
    AnalyzeTextResponse,
    HistoryResponse,
    ParseDemoRequest,
    ParseDemoResponse,
)

app = FastAPI(
    title="Competitor Monitoring Assistant",
    description="Multimodal assistant for competitor monitoring (text + images).",
    version="0.1.1",
)

# Allow local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze_text", response_model=AnalyzeTextResponse)
async def analyze_text(payload: AnalyzeTextRequest) -> AnalyzeTextResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        analysis = openai_client.analyze_competitor_text(payload.text)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist in history (store only a short preview of text)
    history_store.add_history_entry(
        "text",
        {
            "input_preview": payload.text[:200],
            "analysis": analysis,
        },
    )

    return AnalyzeTextResponse(**analysis)


@app.post("/analyze_image", response_model=AnalyzeImageResponse)
async def analyze_image(
    file: UploadFile = File(...),
    comment: Optional[str] = Form(None),
) -> AnalyzeImageResponse:
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 МБ
    
    logger.info(f"[analyze_image] Начало анализа изображения: filename={file.filename}, content_type={file.content_type}")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"[analyze_image] Неподдерживаемый тип файла: {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail=f"Поддерживаются только изображения. Получен тип: {file.content_type}"
        )

    try:
        logger.info("[analyze_image] Читаю файл...")
        image_bytes = await file.read()
        
        if not image_bytes:
            logger.warning("[analyze_image] Файл пуст")
            raise HTTPException(status_code=400, detail="Загруженное изображение пусто.")
        
        file_size = len(image_bytes)
        logger.info(f"[analyze_image] Файл прочитан, размер: {file_size} байт ({file_size / 1024 / 1024:.2f} МБ)")
        
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"[analyze_image] Файл слишком большой: {file_size} байт (максимум {MAX_FILE_SIZE} байт)")
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой ({file_size / 1024 / 1024:.2f} МБ). Максимальный размер: 5 МБ."
            )
        
        logger.info("[analyze_image] Отправляю изображение в OpenAI для анализа...")
        analysis = openai_client.analyze_competitor_image(
            image_bytes=image_bytes, mime_type=file.content_type
        )
        logger.info("[analyze_image] Анализ от OpenAI получен успешно")
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[analyze_image] Ошибка при анализе изображения: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка анализа изображения: {exc}"
        )

    try:
        logger.info("[analyze_image] Сохранение в историю...")
        history_store.add_history_entry(
            "image",
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": file_size,
                "comment": comment,
                "analysis": analysis,
            },
        )
        logger.info("[analyze_image] Анализ завершён успешно")
    except Exception as exc:
        logger.warning(f"[analyze_image] Ошибка при сохранении в историю (не критично): {exc}")

    return AnalyzeImageResponse(**analysis)


def _extract_basic_content(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    p_tag = soup.find("p")
    return {
        "title": title_tag.get_text(strip=True) if title_tag else None,
        "h1": h1_tag.get_text(strip=True) if h1_tag else None,
        "first_paragraph": p_tag.get_text(strip=True) if p_tag else None,
    }


def _fetch_html_with_selenium(url: str, timeout: int = 30) -> str:
    """
    Use headless Chrome via Selenium to fetch fully rendered HTML.
    This is more robust for competitor sites with client‑side rendering.
    """
    logger.info(f"[Selenium] Начинаю загрузку страницы: {url}")
    
    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = None
    try:
        logger.info("[Selenium] Устанавливаю ChromeDriver...")
        service = Service(ChromeDriverManager().install())
        logger.info("[Selenium] Запускаю headless Chrome...")
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(timeout)
        logger.info(f"[Selenium] Открываю URL (таймаут {timeout}с)...")
        
        driver.get(url)
        logger.info("[Selenium] Страница загружена, ожидаю рендеринг контента...")
        
        # Ждём, пока страница полностью загрузится
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            logger.info("[Selenium] Страница полностью загружена")
        except TimeoutException:
            logger.warning("[Selenium] Таймаут ожидания загрузки страницы, но продолжаю...")
        
        html = driver.page_source
        html_size = len(html)
        logger.info(f"[Selenium] HTML получен, размер: {html_size} символов")
        
        if html_size < 100:
            logger.warning(f"[Selenium] Получен очень маленький HTML ({html_size} символов), возможно страница не загрузилась")
        
        return html
        
    except TimeoutException as exc:
        logger.error(f"[Selenium] Таймаут при загрузке страницы {url}: {exc}")
        raise HTTPException(
            status_code=408,
            detail=f"Таймаут загрузки страницы (превышено {timeout} секунд). Возможно, сайт слишком медленный или блокирует автоматические запросы."
        )
    except WebDriverException as exc:
        logger.error(f"[Selenium] Ошибка WebDriver: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка запуска браузера: {exc}",
        )
    except Exception as exc:
        logger.error(f"[Selenium] Неожиданная ошибка при загрузке {url}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Неожиданная ошибка при загрузке страницы: {exc}",
        )
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("[Selenium] Браузер закрыт")
            except Exception as exc:
                logger.warning(f"[Selenium] Ошибка при закрытии браузера: {exc}")


@app.post("/parse_demo", response_model=ParseDemoResponse)
async def parse_demo(payload: ParseDemoRequest) -> ParseDemoResponse:
    url = str(payload.url)
    logger.info(f"[parse_demo] Начало анализа URL: {url}")

    try:
        logger.info("[parse_demo] Этап 1/4: Загрузка HTML через Selenium...")
        html = _fetch_html_with_selenium(url)
        logger.info("[parse_demo] Этап 1/4: HTML успешно получен")
    except HTTPException:
        # уже содержит корректный статус/сообщение
        logger.error(f"[parse_demo] Ошибка HTTP при загрузке {url}")
        raise
    except Exception as exc:
        logger.error(f"[parse_demo] Неожиданная ошибка при загрузке {url}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка загрузки страницы через Selenium: {exc}"
        )

    try:
        logger.info("[parse_demo] Этап 2/4: Извлечение контента (title, h1, p)...")
        extracted = _extract_basic_content(html)
        logger.info(
            f"[parse_demo] Извлечено: title='{extracted.get('title', 'N/A')[:50]}...', "
            f"h1='{extracted.get('h1', 'N/A')[:50]}...', "
            f"p='{extracted.get('first_paragraph', 'N/A')[:50]}...'"
        )
    except Exception as exc:
        logger.error(f"[parse_demo] Ошибка при парсинге HTML: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка извлечения контента из HTML: {exc}"
        )

    def _is_access_denied_page(data: dict) -> bool:
        text = " ".join(
            part
            for part in [
                data.get("title") or "",
                data.get("h1") or "",
                data.get("first_paragraph") or "",
            ]
            if part
        ).lower()
        if not text:
            return False
        markers = [
            "access denied",
            "access is denied",
            "доступ запрещен",
            "доступ ограничен",
            "нет доступа",
        ]
        return any(m in text for m in markers)

    # Если сайт вернул страницу Access Denied — не отправляем это в модель,
    # а сразу возвращаем понятный ответ.
    if _is_access_denied_page(extracted):
        logger.warning("[parse_demo] Обнаружена страница Access Denied, пропускаю анализ через OpenAI")
        analysis = {
            "strengths": [],
            "weaknesses": [
                "Контент сайта недоступен: сервер вернул страницу 'Access Denied'.",
                "Невозможно оценить оффер и коммуникацию без доступа к реальному содержимому страницы.",
            ],
            "unique_offers": [],
            "recommendations": [
                "Попробуйте открыть сайт вручную в браузере и убедиться, что доступ не ограничен по IP или региону.",
                "Для автоматического мониторинга конкурентов может потребоваться прокси/другой регион доступа или согласованный API.",
            ],
        }
    else:
        # Prepare text for analysis
        logger.info("[parse_demo] Этап 3/4: Подготовка текста для анализа...")
        text_parts: List[str] = []
        if extracted["title"]:
            text_parts.append(f"TITLE: {extracted['title']}")
        if extracted["h1"]:
            text_parts.append(f"H1: {extracted['h1']}")
        if extracted["first_paragraph"]:
            text_parts.append(f"FIRST PARAGRAPH: {extracted['first_paragraph']}")

        joined_text = "\n".join(text_parts) if text_parts else "No content found."
        logger.info(f"[parse_demo] Подготовленный текст для анализа ({len(joined_text)} символов): {joined_text[:200]}...")

        try:
            logger.info("[parse_demo] Этап 4/4: Отправка в OpenAI для анализа...")
            analysis = openai_client.analyze_competitor_text(joined_text)
            logger.info("[parse_demo] Анализ от OpenAI получен успешно")
        except Exception as exc:  # pragma: no cover
            logger.error(f"[parse_demo] Ошибка при анализе через OpenAI: {exc}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка анализа через OpenAI: {exc}"
            )

    try:
        logger.info("[parse_demo] Сохранение в историю...")
        history_store.add_history_entry(
            "parse_demo",
            {
                "url": url,
                "extracted": extracted,
                "analysis": analysis,
            },
        )
        logger.info("[parse_demo] Анализ завершён успешно, возвращаю результат")
    except Exception as exc:
        logger.warning(f"[parse_demo] Ошибка при сохранении в историю (не критично): {exc}")

    return ParseDemoResponse(
        url=payload.url,
        title=extracted["title"],
        h1=extracted["h1"],
        first_paragraph=extracted["first_paragraph"],
        analysis=analysis,
    )


@app.get("/history", response_model=HistoryResponse)
async def history() -> HistoryResponse:
    items = history_store.get_history()
    return HistoryResponse(items=items)


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """
    Simple health endpoint with minimal description.
    The actual UI is served from the separate frontend mini-page.
    """
    return JSONResponse(
        {
            "status": "ok",
            "message": "Competitor Monitoring Assistant backend is running.",
        }
    )

