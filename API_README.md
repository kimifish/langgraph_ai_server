# AI Server API Documentation

## Обзор

AI Server предоставляет REST API для взаимодействия с различными AI моделями через унифицированный интерфейс. Сервер поддерживает многопоточные разговоры, персонализацию пользователей и гибкую настройку поведения AI.

## Быстрый старт

### Базовый запрос

```bash
curl "http://localhost:8000/common?message=Привет, как дела?"
```

### Ответ

```json
{
    "answer": "Привет! У меня всё отлично, спасибо что спросил. Чем могу помочь?",
    "thread_id": "Undefined_123456789",
    "llm_used": "common",
    "error": ""
}
```

## Эндпоинты

### 1. Чат с AI - `GET /{llm}`

Основной эндпоинт для общения с AI.

**Параметры:**
- `llm` (path): Идентификатор LLM (common, gpt-4, claude, etc.)
- `message` (query): Сообщение для AI
- `thread_id` (query, optional): ID потока для продолжения разговора
- `user` (query, optional): Идентификатор пользователя
- `location` (query, optional): Локация пользователя
- `additional_instructions` (query, optional): Дополнительные инструкции
- `llm_to_use` (query, optional): Переопределить LLM для этого запроса

**Примеры использования:**

#### Простой чат
```bash
curl "http://localhost:8000/common?message=Расскажи о Python"
```

#### Продолжение разговора
```bash
# Первый запрос
curl "http://localhost:8000/gpt-4?message=Что такое машинное обучение?&user=john"

# Ответ содержит thread_id, используем его для продолжения
curl "http://localhost:8000/gpt-4?message=Приведи примеры&thread_id=john_123456789&user=john"
```

#### С дополнительными инструкциями
```bash
curl "http://localhost:8000/claude?message=Объясни квантовую физику&additional_instructions=Используй простые аналогии, избегай математики"
```

### 2. Проверка здоровья - `GET /health`

Простая проверка работоспособности сервера.

**Ответ:**
```json
{
    "status": "ok",
    "phase": "refactoring"
}
```

### 3. Детальная проверка здоровья - `GET /health/detailed`

Подробная информация о состоянии системы.

**Ответ:**
```json
{
    "status": "ok",
    "timestamp": "2025-09-25T22:26:58.612931",
    "version": "0.3.0",
    "phase": "refactoring",
    "system": {
        "platform": "Linux",
        "python_version": "3.10.12",
        "cpu_count": 8,
        "memory_total": 17179869184,
        "memory_available": 8589934592
    },
    "user_configurations": {
        "total": 5,
        "status": "ok"
    },
    "llm_services": {
        "status": "active",
        "available_llms": ["common", "gpt-4", "claude"]
    }
}
```

### 4. Метрики системы - `GET /metrics`

Метрики производительности для мониторинга.

**Ответ:**
```json
{
    "timestamp": "2025-09-25T22:26:58.612931",
    "process": {
        "pid": 12345,
        "cpu_percent": 15.2,
        "memory_mb": 256.7,
        "threads": 8,
        "uptime_seconds": 3600.5
    },
    "system": {
        "cpu_percent": 45.3,
        "memory_percent": 67.8,
        "disk_usage": 23.4
    },
    "conversations": {
        "total_users": 5,
        "active_conversations": 2
    }
}
```

## Доступные LLM

Сервер поддерживает различные AI модели через разных провайдеров:

- `common` - Основная модель (обычно Gemini)
- `gpt-4` - GPT-4 от OpenAI
- `claude` - Claude от Anthropic
- `deepseek` - DeepSeek модели
- И другие, настроенные в конфигурации

## Управление разговорами

### Thread ID
Каждый разговор имеет уникальный `thread_id`. Если не указан, создается автоматически.
Используйте один `thread_id` для продолжения разговора.

### Пользователи
Параметр `user` позволяет персонализировать ответы и сохранять контекст между сессиями.

### Локация
Параметр `location` помогает AI давать более релевантные ответы с учетом географического контекста.

## Обработка ошибок

Все эндпоинты возвращают структурированные ошибки:

```json
{
    "answer": "",
    "thread_id": "temp",
    "llm_used": "common",
    "error": "Описание ошибки"
}
```

## Интерактивная документация

Полная интерактивная документация с примерами доступна по адресам:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Мониторинг

Для мониторинга используйте:
- `/health` - для проверки доступности
- `/health/detailed` - для диагностики
- `/metrics` - для метрик производительности

## Примеры интеграции

### Python
```python
import requests

def chat_with_ai(message, thread_id=None):
    params = {"message": message}
    if thread_id:
        params["thread_id"] = thread_id

    response = requests.get("http://localhost:8000/common", params=params)
    return response.json()

# Пример использования
result = chat_with_ai("Привет!")
print(result["answer"])

result2 = chat_with_ai("Расскажи подробнее", result["thread_id"])
print(result2["answer"])
```

### JavaScript/Node.js
```javascript
const axios = require('axios');

async function chat(message, threadId = null) {
    const params = { message };
    if (threadId) params.thread_id = threadId;

    const response = await axios.get('http://localhost:8000/common', { params });
    return response.data;
}

// Пример
chat("Hello AI").then(result => {
    console.log(result.answer);
    return chat("Tell me more", result.thread_id);
}).then(result2 => {
    console.log(result2.answer);
});
```
