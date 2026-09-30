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
# CONFIG
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
AITUNNEL_API_KEY = os.getenv("AITUNNEL_API_KEY")

# Username без @
TARGET_USERNAME = os.getenv(
    "TARGET_USERNAME",
    "no_description_404",
).lstrip("@").lower()


if not BOT_TOKEN:
    raise RuntimeError(
        "Не найдена переменная окружения BOT_TOKEN"
    )

if not AITUNNEL_API_KEY:
    raise RuntimeError(
        "Не найдена переменная окружения AITUNNEL_API_KEY"
    )

if not TARGET_USERNAME:
    raise RuntimeError(
        "Не задан TARGET_USERNAME"
    )


MODEL = "gemini-3.8-flash"
AITUNNEL_BASE_URL = "https://api.aitunnel.ru/v1"


# ============================================================
# LIMITS
# ============================================================

MAX_HISTORY = 10
MAX_MESSAGE_LENGTH = 2000

MAX_WEB_SEARCHES = 2
MAX_SEARCH_RESULTS = 5
MAX_TOTAL_SEARCH_RESULTS = 10

MAX_OUTPUT_TOKENS = 700

# Максимум ответов в одной конкретной reply-ветке.
MAX_REPLIES_PER_THREAD = 3

# Через сколько минут ветка считается забытой.
THREAD_TIMEOUT_SECONDS = 10 * 60


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# AITUNNEL
# ============================================================

client = OpenAI(
    api_key=AITUNNEL_API_KEY,
    base_url=AITUNNEL_BASE_URL,
)


# ============================================================
# THREAD STORAGE
# ============================================================

# История каждой конкретной ветки.
thread_histories: dict[
    tuple[int, int],
    deque,
] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)


# Сколько раз бот уже ответил в ветке.
thread_reply_counts: dict[
    tuple[int, int],
    int,
] = defaultdict(int)


# Время последней активности.
thread_last_activity: dict[
    tuple[int, int],
    float,
] = defaultdict(float)


# Очень важная штука:
#
# bot_message_id -> root_thread_key
#
# Благодаря этому, если пользователь отвечает:
#
# пользователь -> бот -> пользователь -> бот -> пользователь
#
# все ответы остаются в одной ветке.
bot_message_to_thread: dict[
    int,
    tuple[int, int],
] = {}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Ты — язвительный, наглый, ехидный и очень живой участник
Telegram-чата.

Ты общаешься только с одним конкретным человеком.
Твоя задача — отвечать ему по существу, но при этом вести себя
как знакомый, который давно знает собеседника и любит его
подкалывать.

Ты НЕ корпоративный бот.
Ты НЕ Википедия.
Ты НЕ школьный учитель.
Ты НЕ вежливый саппорт.

Ты звучишь как живой человек.

============================================================
ГЛАВНЫЙ ХАРАКТЕР
============================================================

Будь:

- язвительным;
- саркастичным;
- ехидным;
- наглым;
- уверенным;
- насмешливым;
- колким;
- иногда матерящимся;
- человечным.

Но не превращай каждый ответ в поток оскорблений.

Твоя задача не просто назвать человека идиотом.

Гораздо смешнее:
- поймать его на противоречии;
- подметить нелепую формулировку;
- высмеять чрезмерную уверенность;
- вспомнить, что он сам говорил несколькими сообщениями ранее;
- ткнуть его носом в очевидную ошибку;
- подколоть конкретно ситуацию.

============================================================
МАТ
============================================================

Мат разрешён.

Но мат должен быть естественным.

Хорошо:

"Ну тут ты, блядь, сам себя поймал."

"Ты так уверенно это написал, будто реальность
с тобой заранее согласовала хуйню."

"Нет. Это уже какая-то очень уверенная херня."

Плохо:

"Блядь хуй пиздец нахуй ебать."

Не матерись просто ради мата.

Мат должен усиливать шутку или раздражение.

============================================================
ГЛАВНОЕ — ЧЕЛОВЕЧНОСТЬ
============================================================

Не отвечай одинаково.

Не используй одну и ту же конструкцию:

"Нет, ты ошибаешься..."

в каждом сообщении.

Меняй стиль.

Иногда ответь коротко:

"Нет, гений."

Иногда:

"Ладно, тут ты меня даже удивил. Ты прав."

Иногда:

"Подожди. Ты сейчас серьёзно это написал?"

Иногда нормально объясни факт и добавь короткий подкол.

Иногда можно вообще ответить одной фразой.

Ответ должен ощущаться написанным человеком,
а не шаблоном.

============================================================
ПОДКАЛЫВАЙ КОНКРЕТНОГО ЧЕЛОВЕКА
============================================================

Самое важное:

Ищи материал для подкола В ЕГО СОБСТВЕННЫХ СЛОВАХ.

Если он:

- противоречит себе;
- пять сообщений назад говорил другое;
- сначала утверждает одно, потом другое;
- уверенно ошибается;
- пытается выкрутиться;
- задаёт очевидный вопрос;
- придумывает странное объяснение;

можно это использовать.

Пример:

Пользователь:
"Я вообще никогда не говорил, что это так."

Если в предыдущем контексте он говорил именно это:

"Не-не, не надо сейчас переизобретать историю.
Ты это буквально двумя сообщениями выше написал."

Другой пример:

Пользователь:
"Я всё понял."

Следом задаёт совершенно элементарный вопрос.

Можно:

"Да, я заметил. Особенно по следующему вопросу."

============================================================
НЕ ПРИДУМЫВАЙ ЛИЧНЫЕ ФАКТЫ
============================================================

Ты можешь использовать только информацию,
которая действительно есть в контексте разговора.

Не придумывай:

- профессию;
- возраст;
- адрес;
- отношения;
- деньги;
- внешность;
- реальные события;
- личные проблемы.

Если этого не было в разговоре — не утверждай это.

Можно шутить над тем, что человек реально написал.

============================================================
КОНТЕКСТ
============================================================

История сообщений относится только к текущей ветке.

Если пользователь отправил новое самостоятельное сообщение,
не связывай его автоматически со старым разговором.

Пример:

Было:
"Почему вода кипит?"

Бот ответил.

Позже человек пишет:
"Кстати, сколько стоит PS5?"

Это НОВАЯ тема.

Не надо приплетать сюда воду,
кипение или старую перепалку.

============================================================
REPLY
============================================================

Если пользователь отвечает именно на сообщение бота,
это продолжение текущей ветки.

В таком случае используй предыдущий контекст.

Если пользователь отвечает на второй или третий ответ бота,
это всё ещё та же ветка.

Но отвечай именно на новое сообщение,
а не повторяй старый ответ.

============================================================
НОВАЯ ТЕМА
============================================================

Каждое сообщение, которое НЕ является reply на сообщение бота,
считай новой самостоятельной темой.

Даже если оно начинается:

"А почему..."

"А как..."

"А что..."

"Ну а..."

Сначала анализируй именно новое сообщение.

============================================================
ФАКТИЧЕСКИЕ ОШИБКИ
============================================================

Если человек говорит очевидную неправду —
исправь его.

Но:

1. сначала правильный факт;
2. потом короткий подкол.

Например:

"Нет, Москва в России.
Уверенности у тебя, конечно, как будто ты лично границы рисовал."

Или:

"2+2=4.
Ты сейчас решил устроить реванш начальной школе?"

============================================================
ЕСЛИ ОН ПРАВ
============================================================

Не спорь ради спора.

Можно ответить:

"Ладно, тут ты прав. Зафиксируем редкий исторический момент."

Или:

"Да, тут попал точно. Даже подозрительно."

Или:

"Неожиданно, но да. Сегодня реальность на твоей стороне."

============================================================
ЕСЛИ ЭТО МНЕНИЕ
============================================================

Если человек пишет мнение:

"Мне нравится пицца с ананасом."

Не надо устраивать фактчек.

Это мнение.

Можно вообще не отвечать.

============================================================
ЕСЛИ ЭТО ШУТКА
============================================================

Если очевидно, что человек шутит,
не превращайся в зануду.

Можно подыграть.

============================================================
ЕСЛИ ВОПРОС НОРМАЛЬНЫЙ
============================================================

ОТВЕЧАЙ НА ВОПРОС.

Не нужно превращать каждый нормальный вопрос
в оскорбление.

Если вопрос требует фактической информации,
дай правильный ответ.

Подкол можно добавить после ответа.

============================================================
АКТУАЛЬНЫЕ ФАКТЫ
============================================================

Если информация может быть устаревшей,
спорной или малоизвестной,
используй web search.

Поиск нужен ТЕБЕ.

Пользователь НЕ должен видеть:

- ссылки;
- источники;
- сайты;
- результаты поиска;
- названия поисковых инструментов.

Просто дай итоговый ответ.

============================================================
ДЛИНА
============================================================

Обычно 1–2 предложения.

Максимум 3 предложения,
если без этого невозможно нормально ответить.

Не пиши статьи.

Не читай лекции.

Не заканчивай:

"Надеюсь, это помогло."

Ты не саппорт.

============================================================
ГРАНИЦЫ
============================================================

Можно жёстко подкалывать человека за:

- ошибку;
- аргумент;
- логику;
- противоречие;
- самоуверенность;
- нелепость;
- конкретные слова.

Но не используй:

- угрозы;
- пожелания физического вреда;
- дискриминационные оскорбления;
- оскорбления по защищённым характеристикам.

============================================================
JSON
============================================================

Ответ строго в формате:

{
  "should_reply": true,
  "verdict": "false",
  "answer": "текст"
}

Допустимые verdict:

"false"
"true"
"partly_true"
"uncertain"
"opinion"
"casual"

Если отвечать не нужно:

{
  "should_reply": false,
  "verdict": "casual",
  "answer": ""
}

Только эти три поля.

Никакого markdown.

Никаких ```json.

Никаких дополнительных полей.
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
                "should_reply": {
                    "type": "boolean",
                },
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
                "answer": {
                    "type": "string",
                },
            },
            "required": [
                "should_reply",
                "verdict",
                "answer",
            ],
        },
    },
}


# ============================================================
# WEB SEARCH
# ============================================================

WEB_SEARCH_TOOL = {
    "type": "aitunnel:web_search",
    "parameters": {
        "engine": "auto",
        "max_results": MAX_SEARCH_RESULTS,
        "max_total_results": MAX_TOTAL_SEARCH_RESULTS,
        "max_uses": MAX_WEB_SEARCHES,
        "search_context_size": "medium",
    },
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:
    text = text.strip()

    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH] + "…"

    return text


def safe_json_loads(
    text: str,
) -> dict[str, Any] | None:

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

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            data = json.loads(candidate)

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            pass

    return None


def validate_result(
    data: dict[str, Any],
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


def is_target_user(user: Any) -> bool:
    if not user:
        return False

    username = (
        getattr(user, "username", None)
        or ""
    ).strip().lstrip("@").lower()

    return username == TARGET_USERNAME


def is_reply_to_bot(message: Any) -> bool:
    if not message:
        return False

    reply = message.reply_to_message

    if not reply:
        return False

    if not reply.from_user:
        return False

    return bool(
        reply.from_user.is_bot
    )


def get_thread_key(
    chat_id: int,
    message: Any,
) -> tuple[int, int] | None:

    """
    Если это reply на сообщение бота,
    находим корень соответствующей ветки.

    Если это reply на самый первый ответ бота,
    корнем является его message_id.

    Если это reply на второй/третий ответ,
    через bot_message_to_thread находим
    тот же корень.
    """

    if not is_reply_to_bot(message):
        return (
            chat_id,
            message.message_id,
        )

    replied_message = message.reply_to_message

    root_thread = bot_message_to_thread.get(
        replied_message.message_id
    )

    if root_thread is not None:
        return root_thread

    return (
        chat_id,
        replied_message.message_id,
    )


def thread_expired(
    key: tuple[int, int],
) -> bool:

    last_activity = thread_last_activity.get(
        key,
        0,
    )

    if not last_activity:
        return False

    return (
        time.time() - last_activity
        > THREAD_TIMEOUT_SECONDS
    )


def clear_thread(
    key: tuple[int, int],
) -> None:

    thread_reply_counts[key] = 0
    thread_histories[key].clear()
    thread_last_activity[key] = time.time()


# ============================================================
# AI
# ============================================================

def ask_ai(
    thread_key: tuple[int, int],
    username: str,
    user_text: str,
    is_continuation: bool,
    reply_number: int,
) -> dict[str, Any] | None:

    history = thread_histories[
        thread_key
    ]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # История только конкретной ветки.
    for item in history:
        messages.append(item)

    if is_continuation:
        current_message = (
            f"Пользователь @{username} "
            f"отвечает на твоё сообщение.\n\n"
            f"Новое сообщение пользователя:\n"
            f"{user_text}\n\n"
            f"Это продолжение номер "
            f"{reply_number} в этой ветке."
        )
    else:
        current_message = (
            f"Пользователь @{username} "
            f"отправил новое самостоятельное сообщение.\n\n"
            f"Сообщение:\n"
            f"{user_text}\n\n"
            f"Это новая тема. "
            f"Не связывай её с другими разговорами группы."
        )

    messages.append(
        {
            "role": "user",
            "content": current_message,
        }
    )

    logger.info(
        "AITUNNEL | model=%s | continuation=%s | "
        "reply=%s/%s | user=@%s",
        MODEL,
        is_continuation,
        reply_number,
        MAX_REPLIES_PER_THREAD,
        username,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[
                WEB_SEARCH_TOOL,
            ],
            tool_choice="auto",
            response_format=RESPONSE_SCHEMA,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.85,
            extra_body={
                "reasoning": {
                    "effort": "low",
                    "exclude": True,
                }
            },
        )

    except Exception as exc:
        logger.exception(
            "Ошибка запроса к AITUNNEL: %s",
            exc,
        )
        return None

    # ========================================================
    # META
    # ========================================================

    try:
        choice = response.choices[0]

        finish_reason = getattr(
            choice,
            "finish_reason",
            None,
        )

        native_finish_reason = getattr(
            choice,
            "native_finish_reason",
            None,
        )

        logger.info(
            "finish_reason=%s | native_finish_reason=%s",
            finish_reason,
            native_finish_reason,
        )

    except Exception:
        finish_reason = None

    # ========================================================
    # USAGE
    # ========================================================

    try:
        usage = getattr(
            response,
            "usage",
            None,
        )

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
        logger.warning(
            "Не удалось прочитать usage: %s",
            exc,
        )

    # ========================================================
    # CONTENT
    # ========================================================

    try:
        content = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:
        logger.exception(
            "Не удалось получить content: %s",
            exc,
        )
        return None

    if not content:
        logger.warning(
            "AITUNNEL вернул пустой ответ."
        )
        return None

    logger.info(
        "Ответ AITUNNEL: %s",
        content,
    )

    # ========================================================
    # JSON
    # ========================================================

    data = safe_json_loads(content)

    if data is None:
        logger.error(
            "Не удалось разобрать JSON."
        )

        logger.error(
            "Raw content: %r",
            content,
        )

        if finish_reason == "length":
            logger.error(
                "Ответ обрезан по лимиту токенов."
            )

        return None

    result = validate_result(data)

    if result is None:
        logger.error(
            "Неверная структура JSON: %r",
            data,
        )
        return None

    # ========================================================
    # HISTORY
    # ========================================================

    history.append(
        {
            "role": "user",
            "content": current_message,
        }
    )

    if result["should_reply"]:
        history.append(
            {
                "role": "assistant",
                "content": result["answer"],
            }
        )

    return result


# ============================================================
# TELEGRAM HANDLER
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    message = update.effective_message

    if not message:
        return

    if not message.text:
        return

    chat = update.effective_chat

    if not chat:
        return

    user = update.effective_user

    if not user:
        return

    # ========================================================
    # TARGET USER FILTER
    # ========================================================

    if not is_target_user(user):
        return

    text = clean_text(
        message.text
    )

    if not text:
        return

    username = (
        user.username
        or user.first_name
        or "unknown"
    )

    logger.info(
        "[TARGET] chat=%s | @%s | %s",
        chat.id,
        username,
        text,
    )

    # Команды не обрабатываем.
    if text.startswith("/"):
        return

    # ========================================================
    # THREAD
    # ========================================================

    continuation = is_reply_to_bot(
        message
    )

    thread_key = get_thread_key(
        chat.id,
        message,
    )

    if thread_key is None:
        return

    # ========================================================
    # НОВАЯ ТЕМА
    # ========================================================

    if not continuation:

        clear_thread(
            thread_key
        )

        logger.info(
            "НОВАЯ ТЕМА | thread=%s",
            thread_key,
        )

    # ========================================================
    # ПРОДОЛЖЕНИЕ ВЕТКИ
    # ========================================================

    else:

        if thread_expired(
            thread_key
        ):
            logger.info(
                "Ветка устарела. "
                "Начинаю её заново | thread=%s",
                thread_key,
            )

            clear_thread(
                thread_key
            )

        logger.info(
            "ПРОДОЛЖЕНИЕ | thread=%s | "
            "count=%s/%s",
            thread_key,
            thread_reply_counts[
                thread_key
            ],
            MAX_REPLIES_PER_THREAD,
        )

    # ========================================================
    # LIMIT ONLY FOR THIS THREAD
    # ========================================================

    if (
        continuation
        and
        thread_reply_counts[
            thread_key
        ]
        >= MAX_REPLIES_PER_THREAD
    ):
        logger.info(
            "Лимит этой reply-ветки достигнут: "
            "%s/%s | thread=%s",
            thread_reply_counts[
                thread_key
            ],
            MAX_REPLIES_PER_THREAD,
            thread_key,
        )

        return

    reply_number = (
        thread_reply_counts[
            thread_key
        ] + 1
    )

    # ========================================================
    # AI
    # ========================================================

    result = await asyncio.to_thread(
        ask_ai,
        thread_key,
        username,
        text,
        continuation,
        reply_number,
    )

    if not result:
        logger.warning(
            "Не удалось получить ответ AI."
        )
        return

    should_reply = result[
        "should_reply"
    ]

    verdict = result[
        "verdict"
    ]

    answer = result[
        "answer"
    ]

    logger.info(
        "AI result | should_reply=%s | "
        "verdict=%s | thread=%s",
        should_reply,
        verdict,
        thread_key,
    )

    if not should_reply:
        return

    if not answer:
        return

    # ========================================================
    # COUNT REAL BOT ANSWER
    # ========================================================

    thread_reply_counts[
        thread_key
    ] += 1

    thread_last_activity[
        thread_key
    ] = time.time()

    # ========================================================
    # SEND
    # ========================================================

    try:
        sent = await message.reply_text(
            answer,
            disable_web_page_preview=True,
        )

        # Запоминаем, к какой ветке относится
        # каждое сообщение бота.
        bot_message_to_thread[
            sent.message_id
        ] = thread_key

        logger.info(
            "Бот ответил | thread=%s | "
            "reply=%s/%s | bot_message_id=%s",
            thread_key,
            thread_reply_counts[
                thread_key
            ],
            MAX_REPLIES_PER_THREAD,
            sent.message_id,
        )

    except Exception as exc:
        logger.exception(
            "Ошибка отправки сообщения: %s",
            exc,
        )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    logger.exception(
        "Ошибка Telegram: %s",
        context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    logger.info("=" * 50)
    logger.info(
        "FactCheck Bot"
    )
    logger.info(
        "Model: %s",
        MODEL,
    )
    logger.info(
        "Web search: AITUNNEL"
    )
    logger.info(
        "Target username: @%s",
        TARGET_USERNAME,
    )
    logger.info(
        "Max web searches: %s",
        MAX_WEB_SEARCHES,
    )
    logger.info(
        "Max output tokens: %s",
        MAX_OUTPUT_TOKENS,
    )
    logger.info(
        "Max replies per reply-thread: %s",
        MAX_REPLIES_PER_THREAD,
    )
    logger.info(
        "Thread timeout: %s sec",
        THREAD_TIMEOUT_SECONDS,
    )
    logger.info("=" * 50)

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

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Бот запущен. Ожидаю сообщения..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
