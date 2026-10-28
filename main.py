        directly_addressed,
    )

    if not result:
        logger.warning("Не удалось получить решение от AI.")
        return

    logger.info(
        "AI decision: should_reply=%s | verdict=%s | target=%s | thread=%s",
        result["should_reply"], result["verdict"], is_target, thread_key,
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

        # Ограничиваем таблицу соответствия, чтобы процесс не рос бесконечно.
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

    logger.exception(
        "Ошибка Telegram: %s",
        context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    logger.info("=" * 40)
    logger.info(
        "FactCheck Bot"
    )
    logger.info(
        "Model: %s",
        MODEL,
    )
    logger.info("API: GPTunneL Chat Completions | Target: @%s", TARGET_USERNAME)
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
