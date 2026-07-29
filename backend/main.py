import os
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Travel Assistant Backend")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YA_API_KEY = os.getenv("YA_API_KEY")
YA_FOLDER_ID = os.getenv("YA_FOLDER_ID")
YA_MODEL_NAME = os.getenv("YA_MODEL_NAME", "yandexgpt-5-lite")

if not YA_API_KEY or not YA_FOLDER_ID:
    logger.error("YA_API_KEY или YA_FOLDER_ID не найдены в .env")
    raise RuntimeError("YA_API_KEY и YA_FOLDER_ID обязательны. Проверьте .env в корне проекта.")


class Participant(BaseModel):
    gender: Optional[str] = None
    age_range: Optional[str] = None
    citizenship: Optional[str] = None
    mobility: Optional[str] = None
    diet_preferences: Optional[str] = None

    @model_validator(mode="after")
    def strip_fields(self):
        for field in self.model_fields:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, value.strip())
        return self


class TravelRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=2048)
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    origin_city: Optional[str] = None
    budget_amount: Optional[float] = None
    budget_currency: str = "RUB"
    participants_count: int = Field(1, ge=1, le=20)
    participants: List[Participant] = []

    @model_validator(mode="after")
    def validate_participants_count(self):
        if len(self.participants) != self.participants_count:
            if len(self.participants) < self.participants_count:
                self.participants += [Participant() for _ in range(self.participants_count - len(self.participants))]
            else:
                self.participants = self.participants[:self.participants_count]
        return self


def format_participant_details(participants: List[Participant]) -> str:
    if not participants:
        return "Нет персональных данных участников."
    parts = []
    for i, p in enumerate(participants, start=1):
        desc = f"Участник {i}"
        details = []
        if p.gender:
            details.append(f"пол: {p.gender}")
        if p.age_range:
            details.append(f"возраст: {p.age_range}")
        if p.citizenship:
            details.append(f"гражданство: {p.citizenship}")
        if p.mobility:
            details.append(f"мобильность: {p.mobility}")
        if p.diet_preferences:
            details.append(f"диета: {p.diet_preferences}")
        if details:
            desc += " (" + ", ".join(details) + ")"
        parts.append(desc)
    return "; ".join(parts)


def build_yandex_payload(req: TravelRequest) -> dict:
    base_prompt = (
        "Ты — ассистент по организации путешествий. Твоя задача — давать конкретные, "
        "практические рекомендации по маршруту, проживанию, транспорту, визам, страховкам "
        "и достопримечательностям. "
        "Важно: не выдумывай факты. Если у тебя нет точных данных о ценах или правилах — "
        "укажи, где пользователь может их найти. "
        "Всегда разделяй варианты на категории «средний бюджет» и «высокий бюджет». "
        "Учитывай параметры поездки: "
    )

    details = []
    if req.departure_date:
        details.append(f"Дата выезда: {req.departure_date}")
    if req.return_date:
        details.append(f"Дата возвращения: {req.return_date}")
    if req.origin_city:
        details.append(f"Город отправления: {req.origin_city}")
    if req.budget_amount is not None:
        details.append(f"Ориентировочный бюджет: {req.budget_amount} {req.budget_currency}")

    participant_text = format_participant_details(req.participants)
    details.append(f"Участники: {participant_text}")

    full_prompt = base_prompt + (", ".join(details) if details else "нет дополнительных параметров") + "."

    messages = [
        {"role": "system", "text": full_prompt},
        {"role": "user", "text": req.user_message}
    ]

    # Правильный формат для новых моделей: gpt://<folder_id>/<model_name>
    model_uri = f"gpt://{YA_FOLDER_ID}/{YA_MODEL_NAME}"
    logger.info(f"Используем modelUri: {model_uri}")

    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "temperature": 0.7,
            "maxTokens": 2000
        },
        "messages": messages
    }
    return payload


def call_yandex_ai(payload: dict) -> str:
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YA_API_KEY}"
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        logger.error(f"Yandex вернул статус {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=f"Yandex AI error: {resp.text}")

    data = resp.json()
    candidates = data.get("result", {}).get("alternatives", [])
    if not candidates:
        raise HTTPException(status_code=502, detail="Пустой ответ от Yandex AI")
    return candidates[0].get("message", {}).get("text", "")


@app.post("/chat")
async def chat(req: TravelRequest):
    try:
        logger.info(f"Получен запрос: user_message={req.user_message[:50]}...")
        payload = build_yandex_payload(req)
        response_text = call_yandex_ai(payload)
        return {"response": response_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Критическая ошибка в /chat: type={type(e).__name__}, msg={str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка бэкенда: {type(e).__name__}: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
