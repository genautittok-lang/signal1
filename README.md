```markdown
# Scalp Signal Bot — LONG & SHORT (for futures)

Коротко:
- Бот сканує ф'ючерсні USDT пари (за замовчуванням Bybit через ccxt).
- Обчислює набір індикаторів і оцінює score для LONG і SHORT.
- Коли score >= CONFIDENCE_THRESHOLD — надсилає сигнал в Telegram з рекомендаціями (margin, leverage, qty, TP, SL).
- Ніяких автоматичних ордерів — тільки сигнали для ручного входу.

Важливо: працює на реальному ринку (за замовчуванням USE_TESTNET=false). Для безпеки не вмикай автотрейд, і починай з RUN_ONCE=true для тесту.

Файли:
- scalp_signal_bot.py — основний скрипт
- requirements.txt
- .env.example
- Procfile (для Railway)
- Dockerfile (опційно)

Як запустити локально:
1. Клонуй репо.
2. Створи віртуальне середовище і встанови deps:
   pip install -r requirements.txt
3. Скопіюй `.env.example` в `.env` та задай TELEGRAM_TOKEN та TELEGRAM_CHAT_ID (numeric id).
4. Встанови бажані параметри (MARGIN_USD, LEVERAGE, TOP_N_BY_VOLUME, CHECK_INTERVAL, CONFIDENCE_THRESHOLD).
5. Для першого тесту постав `RUN_ONCE=true`.
6. Запусти:
   python scalp_signal_bot.py

ENV приклади (в .env):
- TELEGRAM_TOKEN=...
- TELEGRAM_CHAT_ID=...
- USE_TESTNET=false
- TOP_N_BY_VOLUME=50
- CHECK_INTERVAL=30
- CHECK_INTERVAL=30
- CONFIDENCE_THRESHOLD=0.80
- COOLDOWN_MINUTES=15
- MARGIN_USD=10
- LEVERAGE=50
- RUN_ONCE=true

Поради:
- TELEGRAM_CHAT_ID: краще numeric id. Якщо не знаєш, додай бота в чат і використовуй інструменти для отримання id, або напиши /start і зчитай chat.id через getUpdates.
- Щоб збирати статистику, перевір файл `signals_log.csv`.
- Якщо хочеш графіки у повідомленнях — встанови CHARTS_ENABLED=true, але це уповільнить роботу.

Безпека:
- Ніколи не комміть приватні API-ключі.
- Цей бот — тільки сигнали. Якщо захочеш автотрейд, потрібне додаткове тестування та перевірки.

Як розгорнути в Railway:
1. Push repo на GitHub і підключи до Railway.
2. Додай Variables у Railway (TELEGRAM_TOKEN, TELEGRAM_CHAT_ID та інші ENV).
3. Procfile використовує `web: python scalp_signal_bot.py`.
4. Для тестування спочатку RUN_ONCE=true.

Якщо хочеш, я можу:
- Додати картки графіків у повідомлення.
- Додати inline кнопки TAKE/IGNORE (потребують polling/webhook та більше логіки).
- Паралелізувати fetch_ohlcv для більш швидкого сканування (асинхронно).
```
