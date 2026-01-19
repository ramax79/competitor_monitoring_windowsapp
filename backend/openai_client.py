import base64
import json
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


# Load variables from .env (if present) at import time.
load_dotenv()

OPENAI_MODEL_TEXT = os.getenv("OPENAI_MODEL_TEXT", "gpt-4o-mini")
OPENAI_MODEL_IMAGE = os.getenv("OPENAI_MODEL_IMAGE", "gpt-4o-mini")


def _get_client() -> OpenAI:
    """
    Create a reusable OpenAI client.

    OPENAI_API_KEY and optional OPENAI_MODEL_* are taken from .env
    or the process environment.
    """
    return OpenAI()


def analyze_competitor_text(text: str) -> Dict[str, Any]:
    """
    Ask the model to analyze competitor text and return a structured JSON response.
    Модель должна отвечать строго на русском языке.
    """
    logger.info(f"[OpenAI] Начинаю анализ текста (длина: {len(text)} символов)")
    logger.debug(f"[OpenAI] Текст для анализа: {text[:200]}...")
    
    try:
        client = _get_client()
        logger.info(f"[OpenAI] Используется модель: {OPENAI_MODEL_TEXT}")

        system_prompt = (
            "Ты маркетинговый аналитик. Твоя задача — разобрать маркетинговый текст "
            "конкурента и вернуть СТРОГО один JSON‑объект со следующими полями:\n"
            "- strengths: список строк с сильными сторонами конкурента;\n"
            "- weaknesses: список строк со слабыми сторонами;\n"
            "- unique_offers: список строк с уникальными предложениями / УТП;\n"
            "- recommendations: список строк с рекомендациями по улучшению стратегии.\n"
            "Отвечай только JSON, без пояснений, все формулировки делай на русском языке."
        )

        logger.info("[OpenAI] Отправляю запрос в OpenAI API...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL_TEXT,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        logger.info("[OpenAI] Ответ от OpenAI получен")

        content = response.choices[0].message.content
        logger.debug(f"[OpenAI] Сырой ответ (первые 200 символов): {content[:200]}...")
        
        try:
            data = json.loads(content)
            logger.info("[OpenAI] JSON успешно распарсен")
        except json.JSONDecodeError as exc:
            logger.warning(f"[OpenAI] Ошибка парсинга JSON, использую fallback: {exc}")
            # Fallback if model responded with non‑JSON – wrap in a generic structure
            data = {
                "strengths": [],
                "weaknesses": [],
                "unique_offers": [],
                "recommendations": [content],
            }
        
        logger.info(f"[OpenAI] Анализ завершён: strengths={len(data.get('strengths', []))}, "
                    f"weaknesses={len(data.get('weaknesses', []))}, "
                    f"unique_offers={len(data.get('unique_offers', []))}, "
                    f"recommendations={len(data.get('recommendations', []))}")
        return data
        
    except Exception as exc:
        logger.error(f"[OpenAI] Критическая ошибка при анализе текста: {exc}", exc_info=True)
        raise


def _encode_image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def analyze_competitor_image(
    image_bytes: bytes, mime_type: str = "image/png"
) -> Dict[str, Any]:
    """
    Ask the model to analyze a competitor image (banner, website, packaging, etc.)
    and return structured JSON with description, marketing insights and visual style evaluation.
    Модель должна отвечать строго на русском языке.
    """
    logger.info(f"[OpenAI] Начинаю анализ изображения (размер: {len(image_bytes)} байт, тип: {mime_type})")
    
    try:
        client = _get_client()
        logger.info(f"[OpenAI] Используется модель: {OPENAI_MODEL_IMAGE}")

        logger.info("[OpenAI] Кодирую изображение в base64...")
        data_url = _encode_image_to_data_url(image_bytes, mime_type=mime_type)
        logger.info(f"[OpenAI] Изображение закодировано (data URL длина: {len(data_url)} символов)")

        system_prompt = (
            "Ты эксперт по маркетингу, брендингу и анимационным креативам. "
            "Пользователь присылает изображение баннера, сайта или упаковки конкурента. "
            "Проанализируй его и верни СТРОГО один JSON‑объект со следующими полями:\n"
            "- description: строка — что изображено на креативе/сайте/упаковке;\n"
            "- marketing_insights: список строк — ключевые маркетинговые инсайты;\n"
            "- visual_style_evaluation: список строк — оценка визуального стиля, тона, сильных/слабых сторон;\n"
            "- design_score: число от 0 до 10 — субъективная оценка качества дизайна и визуального стиля (чем выше, тем лучше);\n"
            "- animation_potential: список строк — идеи, как этот креатив можно развить в анимацию/видео, какие элементы лучше анимировать.\n"
            "Отвечай только JSON, без пояснений, все формулировки делай на русском языке. "
            "design_score обязательно должен быть числом (например, 7.5)."
        )

        logger.info("[OpenAI] Отправляю запрос в OpenAI API (Vision)...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL_IMAGE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Проанализируй это изображение конкурента.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
        )
        logger.info("[OpenAI] Ответ от OpenAI получен")

        content = response.choices[0].message.content
        logger.debug(f"[OpenAI] Сырой ответ (первые 200 символов): {content[:200]}...")
        
        try:
            data = json.loads(content)
            logger.info("[OpenAI] JSON успешно распарсен")
        except json.JSONDecodeError as exc:
            logger.warning(f"[OpenAI] Ошибка парсинга JSON, использую fallback: {exc}")
            data = {
                "description": "",
                "marketing_insights": [],
                "visual_style_evaluation": [content],
                "design_score": 0.0,
                "animation_potential": [],
            }

        # Гарантируем наличие новых полей даже при частичных ответах.
        if "design_score" not in data:
            logger.warning("[OpenAI] design_score отсутствует в ответе, устанавливаю 0.0")
            data["design_score"] = 0.0
        if "animation_potential" not in data:
            logger.warning("[OpenAI] animation_potential отсутствует в ответе, устанавливаю []")
            data["animation_potential"] = []

        logger.info(f"[OpenAI] Анализ изображения завершён: design_score={data.get('design_score')}, "
                    f"insights={len(data.get('marketing_insights', []))}, "
                    f"animation_ideas={len(data.get('animation_potential', []))}")
        return data
        
    except Exception as exc:
        logger.error(f"[OpenAI] Критическая ошибка при анализе изображения: {exc}", exc_info=True)
        raise

