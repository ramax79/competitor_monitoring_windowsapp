from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl


class AnalyzeTextRequest(BaseModel):
    text: str


class AnalyzeTextResponse(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    unique_offers: List[str]
    recommendations: List[str]


class AnalyzeImageResponse(BaseModel):
    description: str
    marketing_insights: List[str]
    visual_style_evaluation: List[str]
    design_score: float
    animation_potential: List[str]


class ParseDemoRequest(BaseModel):
    url: HttpUrl


class ParseDemoResponse(BaseModel):
    url: HttpUrl
    title: Optional[str]
    h1: Optional[str]
    first_paragraph: Optional[str]
    analysis: Dict[str, Any]


class HistoryItem(BaseModel):
    timestamp: str
    type: str
    payload: Dict[str, Any]


class HistoryResponse(BaseModel):
    items: List[HistoryItem]

