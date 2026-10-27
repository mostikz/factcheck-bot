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
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
TOKENN_API_KEY = os.getenv("TOKENN_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("Не найден BOT_TOKEN в .env")

if not TOKENN_API_KEY:
    raise RuntimeError("Не найден TOKENN_API_KEY в .env")

MODEL = os.getenv("MODEL", "gpt-6.1-sol").strip() or "gpt-6.1-sol"
TOKENN_BASE_URL = "https://api.tokenn.pro/v1"

MAX_HISTORY = 8
MAX_MESSAGE_LENGTH = 2000

MAX_WEB_SEARCHES = 2
MAX_SEARCH_RESULTS = 5
MAX_TOTAL_SEARCH_RESULTS = 10

MAX_OUTPUT_TOKENS = 700

MAX_REPLIES_PER_THREAD = 3
THREAD_TIMEOUT_SECONDS = 10 * 60

TARGET_USERNAME = (
    os.getenv("TARGET_USERNAME", "gonzo_alive")
    .lstrip("@")
    .lower()
)

MAX_PROCESSED_MESSAGES = 3000
MIN_SECONDS_BETWEEN_BOT_REPLIES = 2.0
MIN_SECONDS_BETWEEN_USER_REPLIES = 4.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# TOKENN API
# ============================================================

client = OpenAI(
    api_key=TOKENN_API_KEY,
    base_url=TOKENN_BASE_URL,
)


# ============================================================
# STATE AND HISTORY
# ============================================================

thread_histories: dict[tuple[int, int], deque] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)

thread_reply_counts: dict[tuple[int, int], int] = defaultdict(int)
thread_last_activity: dict[tuple[int, int], float] = defaultdict(float)

# Короткая история каждого пользователя в конкретном чате.
user_histories: dict[tuple[int, int], deque] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY * 2)
)

# Связь ответа бота с корнем ветки.
bot_message_roots: dict[tuple[int, int], tuple[int, int]] = {}

# Дедупликация Telegram-сообщений.
processed_message_keys: set[tuple[int, int]] = set()
processed_message_order: deque = deque()

# Защита от слишком частых ответов.
last_bot_reply_at = 0.0
last_user_reply_at: dict[tuple[int, int], float] = defaultdict(float)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Ты — язвительный, умный, живой участник русскоязычного Telegram-чата и фактчекер.
Твоя главная задача — отвечать пользователю @gonzo_alive, а другим участникам —
только когда они явно обращаются к тебе.

ГЛАВНОЕ ПРАВИЛО: МОЛЧАНИЕ ЛУЧШЕ БЕСПОЛЕЗНОГО ОТВЕТА.
Для каждого сообщения сначала реши, заслуживает ли оно реакции.
Не отвечай на обычную болтовню, приветствия, мемы, шутки, субъективные вкусы,
верные общеизвестные утверждения и сообщения без содержательного повода.
Не выдумывай ошибку только ради сарказма.

КОГДА ОТВЕЧАТЬ:
- Пользователь задал тебе прямой вопрос или явно упомянул тебя — ответь по существу.
- @gonzo_alive высказал проверяемое фактическое утверждение, которое ложно,
  существенно неточно, сомнительно или содержит логическое противоречие — проверь
  и опровергни, если для этого есть достаточные основания.
- @gonzo_alive повторяет уже разобранный аргумент — используй контекст и не копируй
  прежнее опровержение дословно.
- Если @gonzo_alive прав или утверждение верно — обычно молчи. Признавай правоту,
  только если это естественно в прямом диалоге.
- Если сообщение другого участника не адресовано тебе напрямую — не вмешивайся,
  даже если там ошибка.

ФАКТЫ:
- Факты важнее подколок. Сначала назови конкретную ошибку и объясни её.
- Используй конкретные проверяемые факты, даты, цифры или примеры, когда уверен.
- Не выдумывай статистику, исследования, цитаты, ссылки и источники.
- Для актуальных/спорных фактов используй доступный веб-поиск, если инструмент
  действительно работает. Если поиска нет или данных недостаточно — прямо обозначь
  неопределённость и не притворяйся, что проверил источники.
- Не считай мнение фактической ошибкой. «Мне больше нравится BMW» — вкус, не факт.
- «2 + 2 = 4» — верное утверждение, ответа не требует.
- «Земля квадратная» — ложное утверждение: Земля близка к сплюснутому сфероиду,
  что подтверждается геодезическими измерениями и спутниковыми наблюдениями.

СТИЛЬ:
- Естественный разговорный русский, уверенный, короткий, человечный.
- Сарказм, ирония, едкие подколы и уместный мат допустимы, но не ради мата.
- Высмеивай ошибку, аргумент, противоречие или самоуверенность, а не защищённые
  характеристики человека. Никаких угроз и пожеланий вреда.
- Не звучать как справочник или стандартный ИИ. Не использовать вступления
  «Давайте разберёмся», «Интересный вопрос», «Как ИИ» и формальные заключения.
- Обычно 1–3 предложения. Для сложного опровержения можно больше, если каждый
  дополнительный фрагмент действительно доказывает тезис.
- Не повторяй одну и ту же заготовку и не заканчивай каждый ответ предложением
  продолжить разговор.

КОНТЕКСТ:
- Используй историю предыдущих сообщений того же пользователя в этом чате.
- Учитывай уже приведённые аргументы, замечай противоречия и не повторяй полностью
  прежние ответы.
- Не путай новую тему с предыдущей.
- Если не хватает данных для уверенного вывода, предпочти молчание или честную
  оговорку, а не уверенную выдумку.

ОТВЕТ СТРОГО В JSON с тремя полями:
{
  "should_reply": true,
  "verdict": "false",
  "answer": "текст ответа"
}

Допустимые verdict: "false", "true", "partly_true", "uncertain", "opinion", "casual".
Если не нужно отвечать, верни:
{"should_reply": false, "verdict": "casual", "answer": ""}
Никакого Markdown и никаких дополнительных полей.
"""


# ============================================================
# RESPONSE SCHEMA
# ============================================================

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
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

RESPONSE_JSON_SCHEMA = RESPONSE_SCHEMA["json_schema"]["schema"]
RESPONSES_WEB_SEARCH_TOOL = {"type": "web_search"}


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH] + "…"
    return text


def safe_json_loads(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def validate_result(data: dict[str, Any]) -> dict[str, Any] | None:
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


def is_reply_to_this_bot(message: Any, bot_id: int) -> bool:
    reply = getattr(message, "reply_to_message", None)
    sender = getattr(reply, "from_user", None) if reply else None
    return bool(sender and sender.id == bot_id)


def is_directly_addressed_to_bot(
    message: Any,
    bot_username: str,
    bot_id: int,
) -> bool:
    if is_reply_to_this_bot(message, bot_id):
        return True

    text = getattr(message, "text", "") or ""
    wanted_mention = f"@{bot_username}".lower() if bot_username else ""

    for entity in getattr(message, "entities", None) or []:
        entity_type = str(getattr(entity, "type", "")).lower()

        if entity_type.endswith("mention") and not entity_type.endswith("text_mention"):
            mention = text[entity.offset:entity.offset + entity.length].lower()
            if wanted_mention and mention == wanted_mention:
                return True

        elif entity_type.endswith("text_mention"):
            mentioned_user = getattr(entity, "user", None)
            if mentioned_user and mentioned_user.id == bot_id:
                return True

    if wanted_mention and re.search(
        re.escape(wanted_mention) + r"\b",
        text,
        re.IGNORECASE,
    ):
        return True

    return False


def get_thread_key(
    chat_id: int,
    message: Any,
    bot_id: int,
) -> tuple[int, int]:
    reply = getattr(message, "reply_to_message", None)
    sender = getattr(reply, "from_user", None) if reply else None

    if sender and sender.id == bot_id:
        mapped = bot_message_roots.get((chat_id, reply.message_id))
        if mapped:
            return mapped
        return (chat_id, reply.message_id)

    return (chat_id, message.message_id)


def is_obvious_smalltalk(text: str) -> bool:
    normalized = re.sub(r"[\s.!?,…]+", " ", text.lower()).strip()

    if not normalized:
        return True

    smalltalk_patterns = (
        r"^(привет|прив|здарова|здравствуйте|доброе утро|добрый день|добрый вечер|"
        r"спокойной ночи|спасибо|спс|пожалуйста|ок|окей|понял|понятно|ясно|"
        r"ага|угу|неа|да|нет|лол|кек|ах(ах)+|ха(ха)+)$",
    )

    if any(re.fullmatch(pattern, normalized) for pattern in smalltalk_patterns):
        return True

    opinion_patterns = (
        r"^(мне (больше )?нравится|мне не нравится|я люблю|я обожаю|я предпочитаю|"
        r"мой любимый|моя любимая|по мне|на мой вкус)\b",
    )

    return any(
        re.search(pattern, normalized)
        for pattern in opinion_patterns
    )


def remember_processed_message(key: tuple[int, int]) -> bool:
    if key in processed_message_keys:
        return False

    processed_message_keys.add(key)
    processed_message_order.append(key)

    while len(processed_message_order) > MAX_PROCESSED_MESSAGES:
        old = processed_message_order.popleft()
        processed_message_keys.discard(old)

    return True


def thread_expired(key: tuple[int, int]) -> bool:
    last_activity = thread_last_activity.get(key, 0)

    if not last_activity:
        return False

    return time.time() - last_activity > THREAD_TIMEOUT_SECONDS


# ============================================================
# AI
# ============================================================

def ask_ai(
    thread_key: tuple[int, int],
    user_key: tuple[int, int],
    username: str,
    user_text: str,
    is_continuation: bool,
    reply_number: int,
    is_target: bool,
    directly_addressed: bool,
) -> dict[str, Any] | None:

    history = user_histories[user_key]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(list(history))

    if is_continuation:
        context_label = (
            f"Это reply-продолжение ветки (ответ номер {reply_number}). "
            "Учитывай сообщения этой беседы, но отвечай только на текущий смысл."
        )
    else:
        context_label = (
            "Это новое сообщение; не связывай его с другой темой без оснований."
        )

    author_rule = (
        "Автор — целевой пользователь @gonzo_alive."
        if is_target
        else (
            "Автор — другой участник, который ЯВНО обратился к боту. "
            "Отвечай только на обращение к боту; не начинай спор сам."
        )
    )

    address_rule = (
        "Сообщение явно адресовано боту."
        if directly_addressed
        else (
            "Сообщение не содержит явного упоминания бота; оцени, "
            "есть ли реальный повод отвечать."
        )
    )

    current_message = (
        f"{author_rule} {address_rule} {context_label}\n\n"
        f"Текст сообщения @{username}:\n{user_text}\n\n"
        "Реши, нужно ли отвечать. Не отвечай, если утверждение верное, "
        "это мнение, шутка, обычная беседа или нет достаточных оснований."
    )

    messages.append({"role": "user", "content": current_message})

    logger.info(
        "Запрос TokenN | model=%s | user=@%s | target=%s | direct=%s | continuation=%s",
        MODEL,
        username,
        is_target,
        directly_addressed,
        is_continuation,
    )

    response = None
    content = None
    finish_reason = None
    used_responses_api = False

    # Сначала пробуем Responses API с веб-поиском, если TokenN его поддерживает.
    try:
        response = client.responses.create(
            model=MODEL,
            input=messages,
            tools=[RESPONSES_WEB_SEARCH_TOOL],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "factcheck_response",
                    "strict": True,
                    "schema": RESPONSE_JSON_SCHEMA,
                }
            },
            max_output_tokens=MAX_OUTPUT_TOKENS,
            reasoning={"effort": "low"},
        )

        used_responses_api = True
        content = getattr(response, "output_text", None)
        status = getattr(response, "status", None)

        if status == "incomplete":
            finish_reason = "length"

        logger.info("TokenN Responses API: status=%s", status)

    except Exception as exc:
        logger.warning(
            "Responses API/web search недоступны; fallback на Chat Completions "
            "без встроенного поиска: %s",
            exc,
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format=RESPONSE_SCHEMA,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.75,
            )

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            content = getattr(choice.message, "content", None)

        except Exception as fallback_exc:
            logger.exception("Ошибка запроса к TokenN: %s", fallback_exc)
            return None

    try:
        usage = getattr(response, "usage", None)

        if usage:
            try:
                usage_dict = usage.model_dump()
            except Exception:
                try:
                    usage_dict = usage.dict()
                except Exception:
                    usage_dict = {}

            logger.info(
                "Usage: %s",
                json.dumps(
                    usage_dict,
                    ensure_ascii=False,
                    default=str,
                ),
            )

    except Exception as exc:
        logger.warning("Не удалось прочитать usage: %s", exc)

    if not content and used_responses_api:
        try:
            parts = []

            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) != "message":
                    continue

                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        value = getattr(part, "text", None)
                        if value:
                            parts.append(value)

            content = "\n".join(parts).strip()

        except Exception as exc:
            logger.warning("Не удалось извлечь текст Responses API: %s", exc)

    if not content:
        logger.warning("TokenN вернул пустой ответ.")
        return None

    data = safe_json_loads(content)

    if data is None:
        logger.error("Не удалось разобрать JSON: %r", content)

        if finish_reason == "length":
            logger.error("Ответ обрезан по лимиту токенов.")

        return None

    result = validate_result(data)

    if result is None:
        logger.error("Неверная структура JSON: %r", data)
        return None

    # Сохраняем историю пользователя даже тогда, когда бот решил промолчать.
    history.append({
        "role": "user",
        "content": f"@{username}: {user_text}",
    })

    if result["should_reply"] and result["answer"]:
        history.append({
            "role": "assistant",
            "content": result["answer"],
        })
    else:
        history.append({
            "role": "assistant",
            "content": (
                "[Решил промолчать: содержательного повода для ответа не было.]"
            ),
        })

    return result


# ============================================================
# TELEGRAM HANDLER
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    global last_bot_reply_at

    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if not message or not message.text or not chat or not user:
        return

    if user.is_bot:
        return

    text = clean_text(message.text)

    if not text or text.startswith("/"):
        return

    message_key = (chat.id, message.message_id)

    if not remember_processed_message(message_key):
        logger.info("Пропущен повторный update: %s", message_key)
        return

    username = (user.username or "").lower()
    display_name = user.username or user.first_name or "unknown"
    is_target = username == TARGET_USERNAME

    bot_id = context.bot.id
    bot_username = context.bot.username or ""

    directly_addressed = is_directly_addressed_to_bot(
        message,
        bot_username,
        bot_id,
    )

    continuation = is_reply_to_this_bot(message, bot_id)

    logger.info(
        "[%s] @%s | target=%s | direct=%s | text=%s",
        chat.id,
        display_name,
        is_target,
        directly_addressed,
        text,
    )

    # Целевой пользователь — основной и единственный, за кем бот наблюдает.
    # Другим участникам бот отвечает только при явном обращении к нему.
    if not is_target and not directly_addressed:
        logger.info(
            "Пропуск: сообщение не от @%s и не адресовано боту.",
            TARGET_USERNAME,
        )
        return

    # Пропускаем очевидную болтовню и субъективные предпочтения.
    if is_target and not directly_addressed and not continuation:
        if is_obvious_smalltalk(text):
            logger.info(
                "Пропуск очевидной болтовни/мнения от целевого пользователя."
            )
            return

    now = time.time()
    user_key = (chat.id, user.id)

    if now - last_bot_reply_at < MIN_SECONDS_BETWEEN_BOT_REPLIES:
        logger.info("Пропуск из-за общего cooldown.")
        return

    if now - last_user_reply_at[user_key] < MIN_SECONDS_BETWEEN_USER_REPLIES:
        logger.info("Пропуск из-за cooldown пользователя.")
        return

    thread_key = get_thread_key(chat.id, message, bot_id)

    if continuation:
        if thread_expired(thread_key):
            logger.info("Reply-ветка устарела; сбрасываю её состояние.")
            thread_reply_counts[thread_key] = 0
            thread_histories[thread_key].clear()
            thread_last_activity[thread_key] = now

        if thread_reply_counts[thread_key] >= MAX_REPLIES_PER_THREAD:
            logger.info("Лимит ответов в ветке достигнут: %s", thread_key)
            return

    else:
        thread_reply_counts[thread_key] = 0
        thread_last_activity[thread_key] = now

    reply_number = thread_reply_counts[thread_key] + 1
    user_history_key = (chat.id, user.id)

    result = await asyncio.to_thread(
        ask_ai,
        thread_key,
        user_history_key,
        display_name,
        text,
        continuation,
        reply_number,
        is_target,
        directly_addressed,
    )

    if not result:
        logger.warning("Не удалось получить решение от AI.")
        return

    logger.info(
        "AI decision: should_reply=%s | verdict=%s | target=%s | thread=%s",
        result["should_reply"],
        result["verdict"],
        is_target,
        thread_key,
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
        last_bot_reply_at = time.time()
        last_user_reply_at[user_key] = last_bot_reply_at

        bot_message_roots[(chat.id, sent.message_id)] = thread_key

        # Не позволяем таблице связей расти бесконечно.
        if len(bot_message_roots) > MAX_PROCESSED_MESSAGES:
            oldest_key = next(iter(bot_message_roots))
            bot_message_roots.pop(oldest_key, None)

        logger.info(
            "Бот ответил | thread=%s | reply=%s/%s | message_id=%s",
            thread_key,
            thread_reply_counts[thread_key],
            MAX_REPLIES_PER_THREAD,
            sent.message_id,
        )

    except Exception as exc:
        logger.exception("Ошибка отправки сообщения: %s", exc)


# ============================================================
# ERROR
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    logger.exception("Ошибка Telegram: %s", context.error)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logger.info("=" * 40)
    logger.info("FactCheck Bot")
    logger.info("Model: %s", MODEL)
    logger.info(
        "API: TokenN | Target: @%s | Responses API with Chat Completions fallback",
        TARGET_USERNAME,
    )
    logger.info("Max web searches: %s", MAX_WEB_SEARCHES)
    logger.info("Max output tokens: %s", MAX_OUTPUT_TOKENS)
    logger.info("Max replies per reply-thread: %s", MAX_REPLIES_PER_THREAD)
    logger.info("Thread timeout: %s sec", THREAD_TIMEOUT_SECONDS)
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

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
