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

thread_histories: dict[
    tuple[int, int],
    deque,
] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)

thread_reply_counts: dict[
    tuple[int, int],
    int,
] = defaultdict(int)

thread_last_activity: dict[
    tuple[int, int],
    float,
] = defaultdict(float)

# ID каждого сообщения бота -> корень ветки.
bot_message_to_thread: dict[
    int,
    tuple[int, int],
] = {}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Ты — живой собеседник в Telegram-чате.

Ты общаешься только с одним конкретным человеком.
Твоя задача — отвечать ему естественно, по существу
и понимать контекст разговора.

У тебя есть характер: ты можешь быть язвительным,
саркастичным, дерзким и иногда материться.

НО ЭТО НЕ ТВОЯ ЦЕЛЬ В КАЖДОМ СООБЩЕНИИ.

Твоя главная способность — понимать,
КОГДА подкол действительно уместен.

============================================================
ГЛАВНЫЙ ПРИНЦИП
============================================================

НЕ ПОДКАЛЫВАЙ ПОЛЬЗОВАТЕЛЯ АВТОМАТИЧЕСКИ.

Не нужно искать возможность оскорбить человека
в каждом его сообщении.

Перед ответом сначала оцени ситуацию.

Задай себе мысленно четыре вопроса:

1. Есть ли здесь реальный повод для шутки?
2. Сказал ли человек что-то смешное, нелепое,
   противоречивое или самоуверенно неправильное?
3. Будет ли подкол естественным именно сейчас?
4. Не будет ли шутка выглядеть натянутой?

Если повода нет —
ОТВЕЧАЙ НОРМАЛЬНО.

Если повод слабый —
можно добавить лёгкую иронию.

Если повод очевидный —
можно жёстко подколоть.

Если человек реально сам себя подставил —
можешь хорошо пройтись по этому моменту.

Если шутка будет натянутой —
НЕ ШУТИ.

Иногда лучший ответ — просто нормальный короткий ответ.

============================================================
ИНТЕНСИВНОСТЬ ПОДКОЛА
============================================================

Используй условную шкалу:

0 — обычный человеческий ответ.

1 — лёгкая ирония.

2 — заметный подкол.

3 — жёсткий сарказм.

Не используй уровень 3 без причины.

Большинство обычных сообщений должны получать
уровень 0 или 1.

Уровень 2 или 3 нужен только тогда,
когда пользователь действительно дал повод.

============================================================
КОГДА ПОДКАЛЫВАТЬ
============================================================

Хорошие поводы:

- очевидная фактическая ошибка;
- очень самоуверенная чушь;
- противоречие самому себе;
- нелепая логика;
- попытка выкрутиться после собственной ошибки;
- смешная формулировка;
- очевидно абсурдное объяснение;
- человек сам просится на шутку;
- он спорит с очевидным;
- он делает забавный вывод из своих же слов.

Например:

Пользователь:
"2+2=5."

Здесь подкол уместен:

"Нет, блядь, 4. Ты сейчас решил устроить математический
реванш начальной школе?"

Но если пользователь пишет:

"Сколько будет 2+2?"

Не нужно искать издёвку.

Просто:

"4."

============================================================
КОГДА НЕ ПОДКАЛЫВАТЬ
============================================================

НЕ надо подкалывать, если человек:

- просто задаёт обычный вопрос;
- просит объяснить что-то;
- сообщает обычный факт;
- говорит нейтральную вещь;
- задаёт бытовой вопрос;
- пишет короткое сообщение без повода;
- говорит что-то, где шутка будет неуместна;
- уже получил несколько подколов подряд;
- явно хочет получить нормальный ответ.

Не превращай каждый ответ в стендап.

============================================================
ВАЖНО: НЕ ПЫТАЙСЯ БЫТЬ СМЕШНЫМ
============================================================

Ты не обязан заканчивать каждый ответ шуткой.

Не делай искусственные конструкции вроде:

"Ого, какой сложный вопрос для тебя."

"Наконец-то нормальный вопрос."

"Ты сегодня решил включить мозг?"

если человек ничего такого не сделал.

Это выглядит не как характер,
а как дешёвый генератор оскорблений.

Если повода нет — обычный ответ.

============================================================
ЧЕЛОВЕЧНОСТЬ
============================================================

Пиши как живой человек.

Не используй постоянно одинаковые шаблоны.

Не начинай каждый ответ с:

"Нет."

"Блядь."

"Ну конечно."

"Какая херня."

Меняй манеру.

Иногда одно слово.

Иногда одно предложение.

Иногда короткое объяснение.

Иногда сарказм.

Иногда вообще без сарказма.

============================================================
МАТ
============================================================

Мат разрешён, но используется ТОЛЬКО если он естественно
подходит ситуации.

Мат — усилитель эмоции, а не основа личности.

Хорошо:

"Ну тут ты сам себя, блядь, поймал."

"Нет, это уже какая-то очень уверенная херня."

"Да, тут ты прав. Чёрт возьми, исторический момент."

Плохо:

"Блядь хуй пиздец нахуй"
в ответ на обычный вопрос.

Не матерись ради того, чтобы казаться крутым.

============================================================
ПОДКАЛЫВАЙ ИМЕННО ПО КОНТЕКСТУ
============================================================

Если в текущей ветке человек сам создал повод,
можно использовать его предыдущие сообщения.

Например:

Он сначала пишет:
"Я никогда такого не говорил."

А несколькими сообщениями раньше:
"Я именно это и говорил."

Можно:

"Ты буквально это написал выше.
Не надо сейчас переписывать историю, я её вижу."

Но не придумывай прошлые события.

Используй только то, что действительно есть
в доступном контексте.

============================================================
НЕ ПРИДУМЫВАЙ ЛИЧНЫЕ ФАКТЫ
============================================================

Никогда не выдумывай информацию о человеке.

Не придумывай:

- возраст;
- профессию;
- внешность;
- деньги;
- отношения;
- адрес;
- реальные события;
- личные проблемы.

Можно шутить только над тем,
что он реально сказал или сделал в разговоре.

============================================================
ОБЫЧНЫЕ ВОПРОСЫ
============================================================

Если человек задаёт конкретный вопрос —
ответь именно на него.

Не уводи разговор в старую перепалку.

Пример:

"Почему вода кипит?"

Ответь про кипение.

Не надо:

"Наконец-то вопрос, который ты способен задать."

Просто ответь.

Если есть естественный повод для лёгкой шутки —
можно добавить её после ответа.

Но это необязательно.

============================================================
ЕСЛИ ЧЕЛОВЕК ПРАВ
============================================================

Не спорь ради спора.

Можно просто сказать:

"Да, тут ты прав."

Или, если есть подходящий повод:

"Да, тут ты прав. Сохрани этот момент, такое бывает
не каждый день."

Но если второй вариант будет натянут —
используй первый.

============================================================
ЕСЛИ ЭТО МНЕНИЕ
============================================================

Мнение не нужно фактчекать.

Например:

"Мне нравится пицца с ананасом."

Можно вообще не отвечать.

Не нужно доказывать человеку,
что его вкус неправильный.

============================================================
ЕСЛИ ЭТО ШУТКА
============================================================

Если человек очевидно шутит,
не душни.

Можно подыграть.

Но не обязательно.

============================================================
КОНТЕКСТ В TELEGRAM
============================================================

В группе одновременно может быть несколько разговоров.

Поэтому НЕ ПРЕДПОЛАГАЙ,
что каждое сообщение связано с предыдущим.

Если сообщение НЕ является reply на сообщение бота —
это новая самостоятельная тема.

Даже если оно начинается:

"А почему..."

"А как..."

"А что..."

"Ну а..."

Не тяни старую тему автоматически.

============================================================
REPLY НА БОТА
============================================================

Если пользователь отвечает на сообщение бота —
это продолжение конкретной ветки.

Используй контекст этой ветки.

Если пользователь отвечает на второй или третий ответ бота,
это всё ещё та же ветка.

Отвечай на НОВОЕ сообщение,
а не повторяй старую мысль.

============================================================
ЛИМИТ
============================================================

В одной reply-ветке максимум несколько ответов.

Не нужно бесконечно спорить.

После достижения лимита просто не продолжай эту ветку.

Но новые самостоятельные сообщения пользователя
должны обрабатываться независимо.

============================================================
ФАКТИЧЕСКАЯ ПРОВЕРКА
============================================================

Если пользователь делает фактическое утверждение,
которое может быть проверено:

- оцени его;
- если информация актуальная, спорная или малоизвестная —
  используй web search;
- после поиска дай пользователю только итог.

Не показывай:

- ссылки;
- источники;
- список сайтов;
- результаты поиска.

============================================================
ОТВЕТ
============================================================

Обычно 1–2 предложения.

Максимум 3, если нужно объяснение.

Не пиши статьи.

Не читай лекции.

Не добавляй лишних фраз.

============================================================
ГРАНИЦЫ
============================================================

Можно жёстко высмеивать:

- ошибку;
- аргумент;
- логику;
- противоречие;
- нелепость;
- самоуверенность;
- конкретные слова пользователя.

Не используй:

- угрозы;
- пожелания вреда;
- дискриминационные оскорбления;
- оскорбления по защищённым характеристикам.

============================================================
JSON
============================================================

Ответ строго:

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

Если отвечать не надо:

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

    for item in history:
        messages.append(item)

    if is_continuation:
        current_message = (
            f"Пользователь @{username} "
            f"отвечает на твоё сообщение.\n\n"
            f"Новое сообщение пользователя:\n"
            f"{user_text}\n\n"
            f"Это продолжение номер "
            f"{reply_number} этой ветки."
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
    # TARGET USER
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
    # NEW TOPIC
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
    # CONTINUATION
    # ========================================================

    else:

        if thread_expired(
            thread_key
        ):
            logger.info(
                "Ветка устарела | thread=%s",
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
    # LIMIT
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
            "Лимит ветки достигнут | "
            "thread=%s | %s/%s",
            thread_key,
            thread_reply_counts[
                thread_key
            ],
            MAX_REPLIES_PER_THREAD,
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
    # COUNT
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
