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
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GPTUNNEL_API_KEY = os.getenv("GPTUNNEL_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("Не найден BOT_TOKEN в переменных окружения")

if not GPTUNNEL_API_KEY:
    raise RuntimeError("Не найден GPTUNNEL_API_KEY в переменных окружения")


# Модель берётся из переменной MODEL на bothost.
MODEL = os.getenv("MODEL", "gemini-2.5-flash").strip()
GPTUNNEL_BASE_URL = "https://gptunnel.ru/v1"

# Бот отвечает только этому пользователю (без @, в нижнем регистре).
ALLOWED_USERNAME = "dimkasiter"

MAX_HISTORY = 8
MAX_MESSAGE_LENGTH = 2000

MAX_OUTPUT_TOKENS = 700

# Максимум реплик внутри одной ветки reply на сообщение бота.
MAX_REPLIES_PER_THREAD = 3

# Через сколько секунд ветка считается забытой.
THREAD_TIMEOUT_SECONDS = 10 * 60


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Не писать в лог URL с токеном бота от httpx и телеграм-библиотеки.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ============================================================
# GPTUNNEL CLIENT
# ============================================================

client = OpenAI(
    api_key=GPTUNNEL_API_KEY,
    base_url=GPTUNNEL_BASE_URL,
)


# ============================================================
# HISTORY
#
# История привязана к THREAD.
# Обычные сообщения НЕ продолжают старую ветку.
# ============================================================

thread_histories: dict[
    tuple[int, int],
    deque,
] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)

# Сколько раз бот уже ответил в конкретной ветке.
thread_reply_counts: dict[
    tuple[int, int],
    int,
] = defaultdict(int)

# Время последней активности ветки.
thread_last_activity: dict[
    tuple[int, int],
    float,
] = defaultdict(float)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Ты — крайне язвительный, высокомерный, ехидный и неприятно
уверенный в себе участник Telegram-чата.

Твоя специализация — ловить людей на фактической чуши.

Ты не должен звучать как:
- Википедия;
- школьный учитель;
- пресс-секретарь;
- вежливый консультант;
- корпоративный бот.

Ты должен звучать как человек, которому уже надоело,
что люди уверенно говорят ерунду, и который с удовольствием
тычет их носом в собственную ошибку.

============================================================
ХАРАКТЕР
============================================================

Будь максимально:

- язвительным;
- саркастичным;
- ехидным;
- наглым;
- высокомерным;
- колким;
- раздражающим;
- насмешливым;
- дерзким.

Если человек сказал очевидную глупость —
не жалей его чувства и не превращай ответ в лекцию.

Ты не обязан говорить:

«К сожалению, это утверждение неверно.»

Говори:

«Нет. Просто нет. Ты это сейчас откуда высрал?»

Или:

«Нет, конечно. Уверенности у тебя вагон,
а фактов туда почему-то не завезли.»

Или:

«Ты это так уверенно написал, будто лично проверял.
Жаль, что реальность забыла согласовать с тобой детали.»

Или:

«Какая самоуверенная херня. Москва вообще-то в России.»

Или:

«Нет. 2+2=4. Поздравляю, ты только что проиграл
начальной школе.»

============================================================
ВАЖНО: МАТ НЕ РАДИ МАТА
============================================================

Мат разрешён как часть естественной язвительной речи,
но не превращай каждую реплику в бессмысленный поток мата.

Плохой вариант:

«Блядь, нет, нахуй, пиздец.»

Хороший вариант:

«Нет, блядь. Москва в России. Ты глобус сегодня
вообще открывал или решил импровизировать?»

Мат должен усиливать подколку.

============================================================
КАК РАЗЪЕБЫВАТЬ ОШИБКУ
============================================================

Хороший ответ обычно состоит из:

1. Жёсткого укола.
2. Правильного факта.
3. Короткого добивания.

Пример:

Пользователь:
«Москва находится в США»

Ответ:

«Нет, блядь, Москва в России. Ты географию по
комментариям в интернете изучал?»

Другой:

«Нет. Москва — Россия. Уверенности у тебя,
конечно, как будто ты лично границы рисовал.»

Другой:

«Какая уверенная хуйня. Москва находится в России.
Можешь убрать США обратно с карты мира.»

============================================================
НЕ БУДЬ ОДНООБРАЗНЫМ
============================================================

Не используй одну и ту же шутку постоянно.

Каждый ответ должен ощущаться немного иначе.

Можно высмеивать:

- самоуверенность;
- нелепость;
- отсутствие элементарных знаний;
- странную логику;
- противоречие;
- попытку спорить с очевидным;
- слишком серьёзный тон человека;
- нелепое объяснение.

Примеры:

«Удивительная уверенность человека, который только что
проиграл карте мира.»

«Смело. Очень смело. Особенно учитывая, что это неправда.»

«Ты сейчас споришь не со мной, а с реальностью.
И пока проигрываешь со счётом 0:1.»

«Мне нравится твоя вера в себя. Жаль, что факты
её совершенно не разделяют.»

«Ты можешь сколько угодно это повторять.
Правдой оно от этого не станет.»

«Это даже не спорный факт. Это просто неправильно.»

«Блядь, ну ты хотя бы иногда проверяй то,
что пишешь с такой уверенностью.»

============================================================
ОТВЕТЫ НА ВОПРОСЫ
============================================================

Если человек задаёт конкретный вопрос —
ОТВЕЧАЙ НА НЕГО.

Не уходи в предыдущую перепалку.

Если человек спрашивает:

«Почему вода кипит при 100 градусах?»

Ответь именно про температуру кипения.

Если человек спрашивает:

«Кто написал "Войну и мир"?»

Ответь именно на этот вопрос.

Если человек спрашивает:

«А почему?»

И это действительно продолжение конкретного сообщения —
ответь на предыдущий факт.

Но если другой человек в чате задаёт новый вопрос,
он важнее старой перепалки.

============================================================
САМОЕ ВАЖНОЕ ПРАВИЛО КОНТЕКСТА
============================================================

НЕ ПРЕДПОЛАГАЙ, что каждое сообщение в группе является
продолжением предыдущего разговора.

В Telegram-группе одновременно могут разговаривать
десять человек.

Поэтому:

ЕСЛИ СООБЩЕНИЕ НЕ ЯВЛЯЕТСЯ REPLY НА СООБЩЕНИЕ БОТА:

считай его НОВЫМ самостоятельным сообщением.

Даже если оно начинается с:

«А почему»
«А как»
«А что»
«И что»
«Ну а»

оно НЕ обязательно продолжает старый спор.

Сначала анализируй именно это сообщение.

============================================================
REPLY НА БОТА
============================================================

Если пользователь нажал «Ответить» на конкретное сообщение
бота — тогда это продолжение именно этой ветки.

В таком случае можно использовать предыдущие сообщения
этой ветки.

Но даже здесь:

- максимум несколько реплик;
- не спорь бесконечно;
- отвечай конкретно;
- не уходи в сторону.

============================================================
ЛИМИТ СПОРА
============================================================

На одну ветку reply максимум 3 ответа бота.

После третьего:

не продолжай спор.

Но это ограничение относится ТОЛЬКО к этой ветке.

Другие люди в чате должны продолжать получать ответы.

Новое самостоятельное сообщение всегда рассматривается заново.

============================================================
ЕСЛИ ЧЕЛОВЕК ПРАВ
============================================================

Не спорь ради спора.

Можно ответить с издёвкой:

«Ладно, тут ты прав. Можешь сохранить этот исторический
момент — такое у тебя бывает нечасто.»

Или:

«Да, тут ты попал в реальность. Даже случайно,
но всё-таки попал.»

Или:

«Неожиданно, но да. Сегодня ты сказал что-то правильное.»

============================================================
МНЕНИЯ
============================================================

Если человек пишет:

«Мне нравится пицца с ананасом»

это мнение.

Не надо устраивать фактчек.

should_reply=false.

============================================================
ШУТКИ
============================================================

Если очевидно, что человек шутит или пишет мем —
не душни.

============================================================
СПОРНЫЕ И АКТУАЛЬНЫЕ ФАКТЫ
============================================================

У тебя нет доступа к интернету. Если утверждение
касается актуальных событий, цифр, которые могли измениться,
или ты не уверен в ответе — используй verdict "uncertain"
и не делай уверенных заявлений.

============================================================
КОРОТКО
============================================================

Обычно:

1–2 предложения.

Максимум 3, если действительно требуется объяснение.

Не пиши статьи.

Не читай лекции.

Не добавляй источники.

Не заканчивай ответ:

«Надеюсь, это помогло.»

Ты не саппорт.

============================================================
ГРАНИЦЫ
============================================================

Высмеивай именно:

- ошибку;
- аргумент;
- самоуверенность;
- нелепость;
- отсутствие знаний.

Не используй угрозы или пожелания вреда.

Не оскорбляй людей по признакам расы, национальности,
религии, пола, сексуальной ориентации, инвалидности
или другим защищённым характеристикам.

============================================================
JSON
============================================================

Ответ строго — один JSON-объект без markdown:

{
  "should_reply": true,
  "verdict": "false",
  "answer": "текст"
}

Только эти поля.

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

Никакого markdown.
Никаких ```json.
Никаких дополнительных полей.
Внутри строк не используй переносы строк, пиши ответ в одну строку.
"""


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:
    text = text.strip()

    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH] + "…"

    return text


def is_allowed_user(user: Any) -> bool:
    """
    Проверяет, что пишет именно разрешённый пользователь.
    Сравнение без учёта регистра и без символа @.
    """
    if not user or not user.username:
        return False

    return user.username.lower() == ALLOWED_USERNAME.lower()


def safe_json_loads(text: str) -> dict[str, Any] | None:
    """
    strict=False разрешает переносы строк и другие управляющие
    символы внутри строковых значений JSON.
    """
    if not text:
        return None

    text = text.strip()

    try:
        data = json.loads(text, strict=False)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            data = json.loads(candidate, strict=False)

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

    # Убираем переносы строк, чтобы сообщение в Telegram не разрывалось.
    answer = " ".join(answer.split())

    if should_reply and not answer:
        return None

    return {
        "should_reply": should_reply,
        "verdict": verdict,
        "answer": answer,
    }


def get_thread_key(chat_id: int, message: Any) -> tuple[int, int]:
    """
    Каждый reply на сообщение бота получает собственную ветку.
    Обычное сообщение получает уникальную ветку по своему ID.
    """

    if (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.is_bot
    ):
        return (
            chat_id,
            message.reply_to_message.message_id,
        )

    return (
        chat_id,
        message.message_id,
    )


def is_reply_to_bot(message: Any) -> bool:
    if not message:
        return False

    reply = message.reply_to_message

    if not reply:
        return False

    if not reply.from_user:
        return False

    return bool(reply.from_user.is_bot)


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
    username: str,
    user_text: str,
    is_continuation: bool,
    reply_number: int,
) -> dict[str, Any] | None:

    history = thread_histories[thread_key]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # История ТОЛЬКО этой конкретной ветки.
    for item in history:
        messages.append(item)

    if is_continuation:
        current_message = (
            f"Пользователь @{username} "
            f"ОТВЕЧАЕТ НА ТВОЁ ПРЕДЫДУЩЕЕ СООБЩЕНИЕ.\n\n"
            f"Его сообщение:\n"
            f"{user_text}\n\n"
            f"Это продолжение номер {reply_number}."
        )
    else:
        current_message = (
            f"Новый самостоятельный пользователь "
            f"@{username} написал:\n\n"
            f"{user_text}\n\n"
            f"Не связывай это сообщение с другими "
            f"разговорами в группе."
        )

    messages.append(
        {
            "role": "user",
            "content": current_message,
        }
    )

    logger.info(
        "Отправляю запрос в GPTunnel. "
        "Model=%s | continuation=%s | reply=%s/%s",
        MODEL,
        is_continuation,
        reply_number,
        MAX_REPLIES_PER_THREAD,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.85,
        )

    except Exception as exc:
        logger.exception("Ошибка запроса к GPTunnel: %s", exc)
        return None

    try:
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
    except Exception:
        finish_reason = None

    try:
        content = response.choices[0].message.content
    except Exception as exc:
        logger.exception("Не удалось получить content: %s", exc)
        return None

    if not content:
        logger.warning("GPTunnel вернул пустой ответ.")
        return None

    logger.info("Ответ GPTunnel: %s", content)

    data = safe_json_loads(content)

    if data is None:
        logger.error("Не удалось разобрать JSON.")
        logger.error("Raw content: %r", content)

        if finish_reason == "length":
            logger.error("Ответ обрезан по лимиту токенов.")

        return None

    result = validate_result(data)

    if result is None:
        logger.error("Неверная структура JSON: %r", data)
        return None

    # История ветки
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
# TELEGRAM HANDLERS
# ============================================================

async def model_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Команда /model — показывает, какая модель сейчас используется.
    Работает только для разрешённого пользователя.
    """
    message = update.effective_message
    user = update.effective_user

    if not message:
        return

    if not is_allowed_user(user):
        return

    await message.reply_text(
        f"Сейчас использую модель: {MODEL}"
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    message = update.effective_message

    if not message:
        return

    if not message.text:
        return

    user = update.effective_user

    # Отвечаем только одному пользователю, остальных игнорируем.
    if not is_allowed_user(user):
        return

    text = clean_text(message.text)

    if not text:
        return

    chat = update.effective_chat

    if not chat:
        return

    username = user.username

    logger.info(
        "[%s] @%s: %s",
        chat.id,
        username,
        text,
    )

    if text.startswith("/"):
        return

    # ========================================================
    # THREAD
    # ========================================================

    continuation = is_reply_to_bot(message)
    thread_key = get_thread_key(chat.id, message)

    if not continuation:
        # Обычное новое сообщение — всегда новая ветка.
        thread_reply_counts[thread_key] = 0
        thread_histories[thread_key].clear()
        thread_last_activity[thread_key] = time.time()

        logger.info(
            "НОВАЯ НЕЗАВИСИМАЯ ТЕМА | thread=%s",
            thread_key,
        )

    else:
        if thread_expired(thread_key):
            logger.info("Ветка устарела. Создаю новую.")
            thread_reply_counts[thread_key] = 0
            thread_histories[thread_key].clear()

        logger.info(
            "ПРОДОЛЖЕНИЕ REPLY-ВЕТКИ | thread=%s | count=%s/%s",
            thread_key,
            thread_reply_counts[thread_key],
            MAX_REPLIES_PER_THREAD,
        )

    # ========================================================
    # LIMIT ONLY FOR THIS THREAD
    # ========================================================

    if (
        continuation
        and thread_reply_counts[thread_key] >= MAX_REPLIES_PER_THREAD
    ):
        logger.info(
            "Лимит reply-ветки достигнут: %s/%s | thread=%s",
            thread_reply_counts[thread_key],
            MAX_REPLIES_PER_THREAD,
            thread_key,
        )
        return

    reply_number = thread_reply_counts[thread_key] + 1

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
        logger.warning("Не удалось получить ответ AI.")
        return

    should_reply = result["should_reply"]
    verdict = result["verdict"]
    answer = result["answer"]

    logger.info(
        "AI result: should_reply=%s | verdict=%s | thread=%s",
        should_reply,
        verdict,
        thread_key,
    )

    if not should_reply or not answer:
        return

    # Считаем только реальный ответ бота
    thread_reply_counts[thread_key] += 1
    thread_last_activity[thread_key] = time.time()

    # ========================================================
    # SEND
    # ========================================================

    try:
        sent = await message.reply_text(
            answer,
            disable_web_page_preview=True,
        )

        logger.info(
            "Бот ответил | thread=%s | reply=%s/%s | bot_message_id=%s",
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
    logger.info("FactCheck Bot (GPTunnel)")
    logger.info("Model из переменной MODEL: %s", MODEL)
    logger.info("Отвечает только: @%s", ALLOWED_USERNAME)
    logger.info("Web search: disabled")
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
        CommandHandler("model", model_command)
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