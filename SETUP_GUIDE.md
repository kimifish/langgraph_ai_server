# Руководство по установке и настройке AI Server

## Быстрая установка

### 1. Установка зависимостей

```bash
# Клонировать репозиторий
git clone <repository-url>
cd ai_server

# Установить зависимости
pip install -r requirements.txt
# или с uv (рекомендуется)
uv sync
```

### 2. Настройка API ключей

Создайте файл `.env` в корне проекта:

```bash
# OpenRouter (рекомендуется для начала)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx

# Дополнительные провайдеры (опционально)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
AIHUBMIX_API_KEY=sk-xxxxxxxxxxxxx
PROXYAPI_API_KEY=sk-xxxxxxxxxxxxx
```

### 3. Первый запуск

```bash
# С uv
uv run ai_server

# Или напрямую
python -m ai_server.main
```

Сервер запустится на `http://localhost:8000`

### 4. Проверка работы

```bash
# Простая проверка
curl http://localhost:8000/health

# Тестовый чат
curl "http://localhost:8000/common?message=Привет!"
```

## Основные настройки

### Конфигурационные файлы

- `config.yaml` - основные настройки сервера
- `prompts.yaml` - промпты для разных AI моделей
- `.env` - секреты и API ключи

### Настройка LLM провайдеров

В `config.yaml` можно настроить доступные модели:

```yaml
agents:
  common:
    model: openrouter_google/gemini-2.5-flash
    temperature: 0.4
    streaming: false

  gpt-4:
    model: openrouter_openai/gpt-4o
    temperature: 0.3
    streaming: true
```

### Настройка пользователей

```yaml
usernames:
  kimifish: "Влад, 42 года"
  fedonator: "Федя, мальчик 11 лет"
  tais: "Тася, девочка 9 лет"

locations:
  F2_LR: "гостиная"
  GF_D: "столовая"
  BF_V: "подвал"
```

## Расширенная настройка

### Добавление новых LLM

1. Добавьте API ключ в `.env`
2. Настройте модель в `config.yaml`
3. Перезапустите сервер

### Настройка логирования

```yaml
logging:
  level: DEBUG  # или INFO, WARNING, ERROR
  format: "%(message)s"
  rich_tracebacks: true
  show_time: true
```

### Настройка сервера

```yaml
server:
  listen_interfaces: 0.0.0.0  # или 127.0.0.1 для локального
  listen_port: 8000
```

## Интеграция с домашними системами

### OpenHAB (умный дом)

```yaml
openhab:
  api_url: http://192.168.1.10:8080/rest/items
  user: ai
  password: your_password
```

### Календарь

```yaml
calendar:
  url: https://your-calendar-server.com/caldav
  username: your_username
  password: your_password
```

### Музыка (MPD)

```yaml
music:
  mpd:
    host: vault.lan
    port: 6600
    password: your_mpd_password
```

## Устранение неполадок

### Сервер не запускается

1. Проверьте API ключи в `.env`
2. Убедитесь что порт 8000 свободен
3. Проверьте логи на ошибки

### AI не отвечает

1. Проверьте интернет соединение
2. Проверьте лимиты API у провайдера
3. Попробуйте другой LLM: `/gpt-4?message=test`

### Высокое потребление памяти

1. Уменьшите количество одновременных пользователей
2. Настройте кеширование в конфиге
3. Перезапустите сервер

## Мониторинг

### Проверка здоровья

```bash
# Базовая проверка
curl http://localhost:8000/health

# Детальная информация
curl http://localhost:8000/health/detailed

# Метрики производительности
curl http://localhost:8000/metrics
```

### Логи

Логи сохраняются в консоль. Для постоянного хранения:

```bash
# Запуск с сохранением логов
uv run ai_server > ai_server.log 2>&1
```

## Безопасность

### Для домашнего использования

- Запускайте на локальной сети (127.0.0.1 или 192.168.x.x)
- Не выставляйте на публичный интернет без дополнительной защиты
- Регулярно обновляйте API ключи

### Переменные окружения

Все чувствительные данные храните в `.env`:

```bash
# API ключи
OPENROUTER_API_KEY=...
DEEPSEEK_API_KEY=...

# Пароли от сервисов
OPENHAB_PASSWORD=...
CALENDAR_PASSWORD=...
DATABASE_PASSWORD=...
```

## Производительность

### Оптимизация

- Используйте локальные модели когда возможно
- Настройте кеширование для часто используемых промптов
- Мониторьте использование памяти и CPU

### Масштабирование

Для большего количества пользователей:
- Увеличьте память сервера
- Настройте connection pooling
- Рассмотрите использование нескольких экземпляров

## Обновление

```bash
# Получить обновления
git pull

# Обновить зависимости
uv sync

# Перезапустить сервер
# (остановите старый процесс и запустите новый)
```

## Поддержка

- Документация API: `http://localhost:8000/docs`
- Логи сервера в консоли
- Проверяйте `/health/detailed` для диагностики
