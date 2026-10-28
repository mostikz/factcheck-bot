
import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

# ============================================================
# CONFIG — ТОЛЬКО GPTUNNEL, БЕЗ TOKENN И AITUNNEL
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GPTUNNEL_API_KEY = os.getenv("GPTUNNEL_API_KEY", "").strip()
GPTUNNEL_BASE_URL = os.getenv(
    "GPTUNNEL_BASE_URL", "https://gptunnel.ru/v1"
).strip().rstrip("/")
MODEL = os.getenv("MODEL", "gpt-6.1-sol").strip() or "gpt-6.1-sol"

TARGET_USERNAME = os.getenv(
    "TARGET_USERNAME", "gonzo_alive"
).strip().lstrip("@").lower()

MAX_HISTORY = 8
MAX_MESSAGE_LENGTH = 2000
MAX_OUTPUT_TOKENS = 1000
MAX_REPLIES_PER_THREAD = 3
THREAD_TIMEOUT_SECONDS = 10 * 60

if not BOT_TOKEN:
    raise RuntimeError("Не найден BOT_TOKEN в переменных окружения")

if not GPTUNNEL_API_KEY:
    raise RuntimeError("Не найден GPTUNNEL_API_KEY в переменных окружения")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# GPTunnel: адрес и авторизация настраиваются здесь.
client = OpenAI(
    api_key=GPTUNNEL_API_KEY,
    base_url=GPTUNNEL_BASE_URL,
    default_headers={"Authorization": GPTUNNEL_API_KEY},
    timeout=60.0,
    max_retries=1,
)

# История и лимиты ведутся отдельно для каждой ветки.
thread_histories: dict[tuple[int, int], deque] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)
thread_reply_counts: dict[tuple[int, int], int] = defaultdict(int)
thread_last_activity: dict[tuple[int, int], float] = defaultdict(float)

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Ты — язвительный, саркастичный участник Telegram-чата.
Твоя специализация — замечать фактическую чушь и уверенно
исправлять её с подколкой. Пиши как живой человек, не как
Википедия, школьный учитель или корпоративный консультант.

СТИЛЬ:
- Сарказм, ирония, ехидство и подколки разрешены.
- Мат допустим, когда естественно усиливает шутку.
- Не повторяй одни и те же формулировки.
- Высмеивай ошибку, аргумент и самоуверенность, а не
  защищённые характеристики человека.
- Никаких угроз и пожеланий вреда.

КОГДА ОТВЕЧАТЬ:
1. Если сообщение содержит вопрос — ответь на вопрос.
2. Если есть явная фактическая ошибка — поправь её.
3. Если утверждение верное — не выдумывай опровержение.
   При необходимости можешь коротко признать правоту с иронией.
4. Мнения, вкусы, обычные реплики и очевидные шутки не нужно
   фактчекать. На них обычно should_reply=false.
5. Не спорь ради спора и не выдумывай факты.
6. Для сомнительных или актуальных утверждений, если не уверен,
   лучше verdict="uncertain", чем уверенная выдумка.
7. Отвечай кратко: обычно 1–2 предложения, максимум 3.
8. Не добавляй ссылки, источники, Markdown или вступления.
9. Учитывай предыдущие сообщения только в переданном контексте.
   Не выдумывай отсутствующие детали.
10. Если пользователь просто продолжает перепалку, отвечай по сути,
    но не раздувай бесконечный спор.

Верни СТРОГО JSON с тремя полями:
{
  "should_reply": true,
  "verdict": "false",
  "answer": "Короткий ответ"
}

Допустимые verdict:
"false", "true", "partly_true", "uncertain", "opinion", "casual".

Если отвечать не нужно:
{
  "should_reply": false,
  "verdict": "casual",
  "answer": ""
}

Никакого Markdown, никаких блоков ```json и дополнительных полей.
"""

# ============================================================
# HELPERS
# ============================================================

def clean_text(value: str) -> str:
    value = value.strip()
    if len(value) > MAX_MESSAGE_LENGTH:
        value = value[:MAX_MESSAGE_LENGTH] + "…"
    return value


def safe_json_loads(value: str) -> dict[str, Any] | None:
    if not value:
        return None

    value = value.strip()

    try:
        data = json.loads(value)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Некоторые модели оборачивают JSON в лишний текст.
    start = value.find("{")
    end = value.rfind("}")

    if start >= 0 and end > start:
        try:
            data = json.loads(value[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def validate_result(data: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(data.get("should_reply"), bool):
        return None

    if data.get("verdict") not in {
        "false", "true", "partly_true",
        "uncertain", "opinion", "casual",
    }:
        return None

    answer = data.get("answer")
    if not isinstance(answer, str):
        return None

    answer = answer.strip()

    if data["should_reply"] and not answer:
        return None

    return {
        "should_reply": data["should_reply"],
        "verdict": data["verdict"],
        "answer": answer,
    }


def is_reply_to_bot(message) -> bool:
    reply = message.reply_to_message
    return bool(
        reply
        and reply.from_user
        and reply.from_user.is_bot
    )


def is_bot_mentioned(message, bot_username: str | None) -> bool:
    if not bot_username:
        return False

    text = message.text or ""
    if f"@{bot_username.lower()}" in text.lower():
        return True

    # Учитываем Telegram entities, если упоминание оформлено
    # как entity, а не обычный текст.
    for entity in message.entities or []:
        if entity.type == "mention":
            mention = text[entity.offset:entity.offset + entity.length]
            if mention.lower() == f"@{bot_username.lower()}":
                return True

    return False


def get_thread_key(chat_id: int, message) -> tuple[int, int]:
    # Ответ на сообщение бота привязываем к ID сообщения бота,
    # на которое ответил пользователь.
    if is_reply_to_bot(message):
        return chat_id, message.reply_to_message.message_id

    # Самостоятельное сообщение — отдельная ветка.
    return chat_id, message.message_id


def thread_expired(key: tuple[int, int]) -> bool:
    last_activity = thread_last_activity.get(key, 0)
    return bool(
        last_activity
        and time.time() - last_activity > THREAD_TIMEOUT_SECONDS
    )


# ============================================================
# GPTUNNEL REQUEST
# ============================================================

def ask_ai(
    thread_key: tuple[int, int],
    username: str,
    user_text: str,
    continuation: bool,
    reply_number: int,
) -> dict[str, Any] | None:

    history = thread_histories[thread_key]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(list(history))

    if continuation:
        current_message = (
            f"Пользователь @{username} отвечает на твоё предыдущее "
            f"сообщение в этой ветке.\n"
            f"Номер ответа в ветке: {reply_number}.\n"
            f"Сообщение пользователя:\n{user_text}"
        )
    else:
        current_message = (
            f"Новое самостоятельное сообщение от @{username}:\n"
            f"{user_text}\n\n"
            "Рассматривай его отдельно от других разговоров группы."
        )

    messages.append({"role": "user", "content": current_message})

    logger.info(
        "Запрос GPTunnel | model=%s | continuation=%s | reply=%s/%s",
        MODEL, continuation, reply_number, MAX_REPLIES_PER_THREAD,
    )

    response = None

    # Некоторые модели/шлюзы не поддерживают json_schema.
    # Пробуем совместимые режимы последовательно.
    attempts = [
        {"type": "json_schema", "json_schema": {
            "name": "factcheck_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "should_reply": {"type": "boolean"},
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "false", "true", "partly_true",
                            "uncertain", "opinion", "casual",
                        ],
                    },
                    "answer": {"type": "string"},
                },
                "required": ["should_reply", "verdict", "answer"],
            },
        }},
        {"type": "json_object"},
        None,
    ]

    last_error = None

    for response_format in attempts:
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.7,
        }

        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Запрос с форматом %s не прошёл: %s",
                response_format.get("type") if response_format else "без формата",
                exc,
            )

    if response is None:
        logger.error("GPTunnel не вернул ответ: %s", last_error)
        return None

    try:
        content = response.choices[0].message.content
    except Exception:
        logger.exception("Не удалось прочитать ответ GPTunnel")
        return None

    if not content:
        logger.warning("GPTunnel вернул пустой ответ")
        return None

    logger.info("Ответ GPTunnel: %s", content)

    data = safe_json_loads(content)
    if data is None:
        logger.error("Ответ модели не является корректным JSON")
        return None

    result = validate_result(data)
    if result is None:
        logger.error("Неверная структура ответа: %r", data)
        return None

    # Сохраняем историю только после успешного разбора ответа.
    history.append({"role": "user", "content": current_message})

    if result["should_reply"]:
        history.append({
            "role": "assistant",
            "content": result["answer"],
        })

    return result


# ============================================================
# TELEGRAM HANDLER
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if not message or not chat or not user or not message.text:
        return

    text = clean_text(message.text)

    if not text or text.startswith("/"):
        return

    username = (user.username or user.first_name or "unknown")
    username_lower = (user.username or "").lower()

    bot_username = context.bot.username
    continuation = is_reply_to_bot(message)
    mentioned = is_bot_mentioned(message, bot_username)

    # Основному адресату анализируем сообщения всегда.
    # Остальным отвечаем только при прямом обращении к боту
    # или reply на сообщение бота.
    is_target = username_lower == TARGET_USERNAME

    if not is_target and not mentioned and not continuation:
        return

    logger.info("[%s] @%s: %s", chat.id, username, text)

    thread_key = get_thread_key(chat.id, message)

    if not continuation:
        thread_reply_counts[thread_key] = 0
        thread_histories[thread_key].clear()
        thread_last_activity[thread_key] = time.time()
    elif thread_expired(thread_key):
        thread_reply_counts[thread_key] = 0
        thread_histories[thread_key].clear()

    if (
        continuation
        and thread_reply_counts[thread_key] >= MAX_REPLIES_PER_THREAD
    ):
        logger.info("Достигнут лимит ответов в ветке %s", thread_key)
        return

    reply_number = thread_reply_counts[thread_key] + 1

    result = await asyncio.to_thread(
        ask_ai,
        thread_key,
        username,
        text,
        continuation,
        reply_number,
    )

    if not result:
        logger.warning("Не удалось получить результат AI")
        return

    logger.info(
        "Результат AI: should_reply=%s | verdict=%s | thread=%s",
        result["should_reply"], result["verdict"], thread_key,
    )

    if not result["should_reply"] or not result["answer"]:
        return

    try:
        sent = await message.reply_text(
            result["answer"],
            disable_web_page_preview=True,
        )

        thread_reply_counts[thread_key] += 1
        thread_last_activity[thread_key] = time.time()

        logger.info(
            "Бот ответил | thread=%s | reply=%s/%s | message_id=%s",
            thread_key,
            thread_reply_counts[thread_key],
            MAX_REPLIES_PER_THREAD,
            sent.message_id,
        )
    except Exception:
        logger.exception("Ошибка отправки сообщения в Telegram")


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    logger.error("Ошибка Telegram: %s", context.error)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logger.info("=" * 40)
    logger.info("FactCheck Bot")
    logger.info("Provider: GPTunnel")
    logger.info("Base URL: %s", GPTUNNEL_BASE_URL)
    logger.info("Model: %s", MODEL)
    logger.info("Target: @%s", TARGET_USERNAME)
    logger.info("Max replies per thread: %s", MAX_REPLIES_PER_THREAD)
    logger.info("=" * 40)

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )
    application.add_error_handler(error_handler)

    logger.info("Бот запущен. Ожидаю сообщения...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
