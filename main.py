import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
AI_API_KEY = (
    os.getenv("AITUNNEL_API_KEY")
    or os.getenv("AI_API_KEY")
    or ""
).strip()
AI_BASE_URL = os.getenv(
    "AI_BASE_URL", "https://api.aitunnel.ru/v1"
).strip()
AI_MODEL = os.getenv("AI_MODEL", "gemini-3.8-flash").strip()

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "2500"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "900"))
MAX_WEB_SEARCHES = int(os.getenv("MAX_WEB_SEARCHES", "2"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_REPLIES_PER_THREAD = int(os.getenv("MAX_REPLIES_PER_THREAD", "5"))
THREAD_TIMEOUT_SECONDS = int(os.getenv("THREAD_TIMEOUT_SECONDS", "900"))
MIN_SECONDS_BETWEEN_UNSOLICITED_REPLIES = float(
    os.getenv("MIN_SECONDS_BETWEEN_UNSOLICITED_REPLIES", "8")
)

if not BOT_TOKEN:
    raise RuntimeError("Не задан BOT_TOKEN в .env")

if not AI_API_KEY:
    raise RuntimeError("Не задан AITUNNEL_API_KEY или AI_API_KEY в .env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sarcastic_bot")

client = OpenAI(
    api_key=AI_API_KEY,
    base_url=AI_BASE_URL,
)

SYSTEM_PROMPT = r"""
Ты — живой участник русскоязычного Telegram-чата:
умный, язвительный, вредный, остроумный, уверенный и иногда неприятный.
У тебя острый язык, но ты не тупой хам и не генератор одинаковых оскорблений.

ТВОЯ ЗАДАЧА:
Поддерживать естественный разговор, отвечать на вопросы, участвовать
в переписке, а когда кто-то уверенно сообщает неверный проверяемый факт —
исправлять его фактами и при случае метко подкалывать.

СТИЛЬ:
- Разговорный русский, естественный ритм, уместный сленг и иногда мат.
- Можно язвить, иронизировать, поддевать самоуверенность и нелепую логику.
- Не будь вежливым ради вежливости и не поддакивай автоматически.
- Не используй шаблонные фразы и одну и ту же шутку из раза в раз.
- Без воды: каждое предложение должно что-то добавлять.
- Простая реплика — короткий ответ; сложный вопрос — нормальное объяснение
  с фактами, примерами и аргументами.
- Не начинай каждый ответ с «Конечно», «Хороший вопрос», «Давай разберёмся».
- Не заканчивай «Надеюсь, это помогло».
- Не злоупотребляй эмодзи, списками и формальными выводами.
- Не утверждай, что ты человек, и не выдумывай личный опыт.

КОГДА ОТВЕЧАТЬ:
Реши, нужно ли вмешиваться в конкретное сообщение.
Верни should_reply=false, если сообщение не требует твоего участия.

1. Если тебя спросили напрямую, упомянули по @username или ответили
   на твоё сообщение — отвечай.
2. Если в чате задан содержательный вопрос, на который ты можешь ответить,
   можешь включиться, даже если тебя не отметили. Не отвечай на каждую
   риторическую фразу, вопрос к конкретному человеку или очевидный мем.
3. Если человек уверенно сообщает проверяемый фактический тезис, который
   существенно неверен, можешь вмешаться и исправить его.
   Делай это только при реальной ошибке, а не из-за мнения или шутки.
4. Не вмешивайся в обычную болтовню, личные предпочтения, приветствия
   и разговоры, где твоё участие лишнее.
5. Если факты неясны или утверждение зависит от контекста, не объявляй его
   ложным только ради подколки. При необходимости используй веб-поиск.
6. Не повторяй исправление, если его уже объяснили и новых аргументов нет.
7. В группе несколько параллельных разговоров. Не связывай новое сообщение
   с чужой старой перепалкой, если это не reply на сообщение бота.

ФАКТЧЕК:
- Различай факт, мнение, шутку, предположение и преувеличение.
- Если человек прав — не выдумывай контраргументы. Можешь признать правоту
  с иронией, но без натужной попытки унизить.
- Если тезис частично верен, укажи, какая часть верна и что не так.
- Для актуальных, спорных или малоизвестных фактов используй веб-поиск,
  если инструмент доступен.
- Никогда не выдумывай статистику, исследования, цитаты, источники или цифры.
- Сначала дай точный ответ или аргумент, затем добавь подколку, если уместно.
- Высмеивай тезис, логику и самоуверенность, а не защищённые характеристики
  человека. Не используй угрозы и не пожелай реального вреда.

КОНТЕКСТ:
- История передаётся только для конкретной ветки ответа на сообщение бота.
- Если сообщение новое и самостоятельное, оценивай его само по себе.
- Если это продолжение reply-ветки, отвечай с учётом истории именно этой ветки.
- Если собеседник сменил тему, отвечай на новую тему.
- Не продолжай бесконечную перепалку: после нескольких ответов ветка остановится.

ФОРМАТ:
Верни только JSON:
{"should_reply": true, "verdict": "false", "answer": "текст"}

Допустимые verdict:
"false", "true", "partly_true", "uncertain", "opinion", "casual".

Если отвечать не нужно:
{"should_reply": false, "verdict": "casual", "answer": ""}

Если отвечать нужно, answer должен содержать обычный текст.
Не оборачивай JSON в Markdown.
"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "chat_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "should_reply": {"type": "boolean"},
                "verdict": {
                    "type": "string",
                    "enum": [
                        "false",
                        "true",
                        "partly_true",
                        "uncertain",
                        "opinion",
                        "casual",
                    ],
                },
                "answer": {"type": "string"},
            },
            "required": ["should_reply", "verdict", "answer"],
        },
    },
}

ENABLE_WEB_SEARCH = (
    os.getenv("ENABLE_WEB_SEARCH", "1").lower()
    not in {"0", "false", "no"}
)

WEB_SEARCH_TOOL = {
    "type": "aitunnel:web_search",
    "parameters": {
        "engine": "auto",
        "max_results": MAX_SEARCH_RESULTS,
        "max_total_results": MAX_SEARCH_RESULTS * 2,
        "max_uses": MAX_WEB_SEARCHES,
        "search_context_size": "medium",
    },
}

histories: dict[tuple[int, int], deque] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)
reply_counts: dict[tuple[int, int], int] = defaultdict(int)
last_activity: dict[tuple[int, int], float] = defaultdict(float)
last_unsolicited_reply: dict[int, float] = defaultdict(float)

bot_username: str | None = None


def clean_text(value: str) -> str:
    value = value.strip()

    if len(value) > MAX_MESSAGE_LENGTH:
        value = value[:MAX_MESSAGE_LENGTH] + "…"

    return value


def is_reply_to_bot(message: Any) -> bool:
    reply = getattr(message, "reply_to_message", None)

    return bool(
        reply
        and reply.from_user
        and reply.from_user.is_bot
    )


def get_thread_key(
    chat_id: int,
    message: Any,
) -> tuple[int, int]:
    if is_reply_to_bot(message):
        return chat_id, message.reply_to_message.message_id

    return chat_id, message.message_id


def thread_expired(key: tuple[int, int]) -> bool:
    last = last_activity.get(key, 0)

    return bool(
        last and time.time() - last > THREAD_TIMEOUT_SECONDS
    )


def is_directly_addressed(message: Any, text: str) -> bool:
    if is_reply_to_bot(message):
        return True

    if bot_username and re.search(
        rf"(?i)(?<!\w)@{re.escape(bot_username)}\b",
        text,
    ):
        return True

    for entity in (getattr(message, "entities", None) or []):
        if getattr(entity, "type", None) == "mention":
            start = entity.offset
            end = start + entity.length
            mention = text[start:end].lower()

            if bot_username and mention == f"@{bot_username.lower()}":
                return True

    return False


def safe_json_loads(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    text = text.strip()

    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            value = json.loads(text[start:end + 1])

            if isinstance(value, dict):
                return value

        except json.JSONDecodeError:
            pass

    return None


def validate_result(
    data: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None

    should_reply = data.get("should_reply")
    verdict = data.get("verdict")
    answer = data.get("answer")

    if not isinstance(should_reply, bool):
        return None

    if verdict not in {
        "false",
        "true",
        "partly_true",
        "uncertain",
        "opinion",
        "casual",
    }:
        return None

    if not isinstance(answer, str):
        return None

    answer = answer.strip()

    if should_reply and not answer:
        return None

    return {
        "should_reply": should_reply,
        "verdict": verdict,
        "answer": answer,
    }


def ask_ai(
    thread_key: tuple[int, int],
    username: str,
    user_text: str,
    direct: bool,
    continuation: bool,
    reply_number: int,
) -> dict[str, Any] | None:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if continuation:
        messages.extend(list(histories[thread_key]))

    current = (
        f"Сообщение от @{username}.\n"
        f"Прямое обращение к боту: {'ДА' if direct else 'НЕТ'}.\n"
        f"Продолжение reply-ветки: {'ДА' if continuation else 'НЕТ'}.\n"
        f"Номер ответа в этой ветке: {reply_number}.\n\n"
        f"Текст сообщения:\n{user_text}\n\n"
        "Реши, нужно ли вмешаться по правилам системного промпта. "
        "Если это просто болтовня или мнение без вопроса и без "
        "фактической ошибки — молчи."
    )

    messages.append({"role": "user", "content": current})

    kwargs: dict[str, Any] = {
        "model": AI_MODEL,
        "messages": messages,
        "response_format": RESPONSE_SCHEMA,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.8,
    }

    if ENABLE_WEB_SEARCH:
        kwargs["tools"] = [WEB_SEARCH_TOOL]
        kwargs["tool_choice"] = "auto"

    kwargs["extra_body"] = {
        "reasoning": {
            "effort": "low",
            "exclude": True,
        }
    }

    try:
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""

        usage = getattr(response, "usage", None)

        if usage:
            logger.info(
                "Token usage: %s",
                usage.model_dump_json()
                if hasattr(usage, "model_dump_json")
                else usage,
            )

        if not content:
            logger.warning("Провайдер вернул пустой ответ")
            return None

        result = validate_result(safe_json_loads(content))

        if result is None:
            logger.error(
                "Не удалось разобрать ответ модели: %r",
                content[:1000],
            )
            return None

        histories[thread_key].append(
            {"role": "user", "content": current}
        )

        if result["should_reply"]:
            histories[thread_key].append(
                {"role": "assistant", "content": result["answer"]}
            )

        return result

    except Exception:
        logger.exception("Ошибка AI API")
        return None


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if (
        not message
        or not message.text
        or not chat
        or not user
        or user.is_bot
    ):
        return

    text = clean_text(message.text)

    if not text or text.startswith("/"):
        return

    username = user.username or user.first_name or "unknown"
    continuation = is_reply_to_bot(message)
    direct = is_directly_addressed(message, text)
    thread_key = get_thread_key(chat.id, message)

    if continuation:
        if thread_expired(thread_key):
            histories[thread_key].clear()
            reply_counts[thread_key] = 0

        if reply_counts[thread_key] >= MAX_REPLIES_PER_THREAD:
            logger.info("Лимит reply-ветки достигнут: %s", thread_key)
            return

    else:
        histories[thread_key].clear()
        reply_counts[thread_key] = 0

    reply_number = reply_counts[thread_key] + 1

    logger.info(
        "Chat=%s user=%s direct=%s continuation=%s text=%r",
        chat.id,
        username,
        direct,
        continuation,
        text[:300],
    )

    result = await asyncio.to_thread(
        ask_ai,
        thread_key,
        username,
        text,
        direct,
        continuation,
        reply_number,
    )

    if not result:
        return

    if not result["should_reply"] or not result["answer"]:
        return

    # Ограничиваем незапрошенные вмешательства, но не ответы на обращения.
    if not direct and not continuation:
        now = time.time()

        if (
            now - last_unsolicited_reply[chat.id]
            < MIN_SECONDS_BETWEEN_UNSOLICITED_REPLIES
        ):
            logger.info(
                "Пропуск ответа из-за cooldown в чате %s",
                chat.id,
            )
            return

        last_unsolicited_reply[chat.id] = now

    try:
        sent = await message.reply_text(
            result["answer"],
            disable_web_page_preview=True,
            reply_to_message_id=message.message_id,
            allow_sending_without_reply=True,
        )

        reply_counts[thread_key] += 1
        last_activity[thread_key] = time.time()

        logger.info(
            "Ответ отправлен | chat=%s thread=%s count=%s/%s "
            "msg=%s verdict=%s",
            chat.id,
            thread_key,
            reply_counts[thread_key],
            MAX_REPLIES_PER_THREAD,
            sent.message_id,
            result["verdict"],
        )

    except Exception:
        logger.exception("Не удалось отправить ответ в Telegram")


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    logger.error(
        "Ошибка Telegram: %s",
        context.error,
        exc_info=context.error,
    )


async def post_init(application: Application) -> None:
    global bot_username

    me = await application.bot.get_me()
    bot_username = me.username

    logger.info("Бот авторизован: @%s", bot_username)


def main() -> None:
    logger.info(
        "Запуск: model=%s base_url=%s web_search=%s max_replies=%s",
        AI_MODEL,
        AI_BASE_URL,
        ENABLE_WEB_SEARCH,
        MAX_REPLIES_PER_THREAD,
    )

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    app.add_error_handler(error_handler)

    logger.info("Бот запущен и ожидает сообщения")

    app.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
