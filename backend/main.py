import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from dotenv import load_dotenv

load_dotenv()

YA_API_KEY = os.getenv("YA_API_KEY")
YA_FOLDER_ID = os.getenv("YA_FOLDER_ID")

if not YA_API_KEY or not YA_FOLDER_ID:
    raise RuntimeError("Не найдены YA_API_KEY или YA_FOLDER_ID в .env")

app = FastAPI()

class Participant(BaseModel):
    age: int
    role: str  # например: adult, child

class TravelRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=2048)
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    origin_city: Optional[str] = None
    budget_amount: Optional[float] = None
    budget_currency: str = "RUB"
    participants_count: int = Field(1, ge=1, le=20)
    participants: List[Participant] = []

def build_yandex_payload(req: TravelRequest) -> dict:
    base_prompt = (
        "Ты — ассистент по организации путешествий по России. Твоя задача — давать конкретные, "
        "практические рекомендации по маршруту, проживанию, транспорту, визам, страховкам "
        "и достопримечательностям. Важно: не выдумывай факты. Если у тебя нет точных данных о ценах "
        "или правилах — укажи, где пользователь может их найти (например, «цены на билеты уточняйте на Aviasales», "
        "'правила въезда — на сайте МИД РФ'). Всегда разделяй варианты на категории «средний бюджет» и «высокий бюджет». "
        "\n\n"
        "Учитывай параметры поездки: город назначения, длительность, общий бюджет, количество участников, "
        "даты поездки. Если в запросе нет города — спроси только один раз. Если город есть — сразу давай готовый план.\n"
        "\n"
        "Строго соблюдай формат ответа:\n"
        "День 1\n- Достопримечательности (2–3 места): название, краткое описание, ориентировочное время на посещение (в минутах).\n- Проживание: 2 варианта (средний бюджет и высокий бюджет) с ориентировочным диапазоном цен за ночь в рублях.\n- Транспорт: как добраться до первой точки + варианты внутри города с примерной стоимостью (автобус/такси/самокат).\n\nДень 2\n...\nДень 3\n...\n\nВ конце добавь блок «Где проверить актуальные данные» и перечисли 2–3 надёжных источника (например, Aviasales, Ostrovok, Яндекс Карты).\n\nНе задавай уточняющих вопросов, если город и бюджет уже указаны. Если каких-то данных не хватает для точных расчётов — используй типичные рыночные диапазоны для этого города и обязательно пометь: «цены ориентировочные, актуальны на 2024 год»."
    )

    details = []
    if req.origin_city:
        details.append(f"Город отправления: {req.origin_city}")
    if req.departure_date and req.return_date:
        details.append(f"Даты: с {req.departure_date} по {req.return_date}")
    elif req.departure_date:
        details.append(f"Дата отправления: {req.departure_date}")
    if req.budget_amount:
        details.append(f"Бюджет: до {req.budget_amount:.0f} {req.budget_currency}")
    if req.participants_count:
        details.append(f"Количество участников: {req.participants_count}")
        if len(req.participants) > 0:
            ages = ", ".join([str(p.age) for p in req.participants])
            details.append(f"Возраст участников: {ages}")

    context_info = "\n".join(details) if details else "Нет дополнительных параметров."

    payload = {
        "modelUri": f"gpt://{YA_FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.7,
            "maxTokens": 2000
        },
        "messages": [
            {"role": "system", "text": base_prompt},
            {"role": "user", "text": f"{context_info}\n\nЗапрос пользователя: {req.user_message}"}
        ]
    }
    return payload

def call_yandex_gpt(req: TravelRequest) -> str:
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {YA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = build_yandex_payload(req)

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["result"]["alternatives"][0]["message"]["text"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка связи с LLM: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: TravelRequest):
    response_text = call_yandex_gpt(req)
    return {"response": response_text}
