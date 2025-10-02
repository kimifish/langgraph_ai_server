# Руководство разработчика AI Server

## Архитектура проекта

### Общая структура

```
ai_server/
├── api/              # HTTP API слой
│   ├── routes.py     # FastAPI роуты
│   ├── schemas.py    # Pydantic модели
│   ├── dependencies.py
│   └── middleware.py
├── services/         # Бизнес логика
│   ├── conversation_service.py  # Управление разговорами
│   ├── llm_service.py          # Работа с LLM
│   ├── tool_service.py         # Инструменты
│   └── user_service.py         # Пользователи
├── models/           # Данные и состояния
│   ├── state.py      # Состояние разговора
│   ├── userconfs.py  # Конфигурации пользователей
│   └── conversation_state.py
├── config.py         # Конфигурация
├── graph.py          # LangGraph workflow
├── llms.py           # LLM фабрики
└── main.py           # Точка входа
```

### Поток данных

1. **HTTP запрос** → `routes.py`
2. **Валидация** → `schemas.py`
3. **Бизнес логика** → `services/`
4. **LLM взаимодействие** → `llms.py` + `graph.py`
5. **Состояние** → `models/`
6. **Ответ** → `routes.py`

## Добавление нового LLM

### 1. Настройка в конфигурации

Добавьте в `config.yaml`:

```yaml
agents:
  new_llm:
    model: provider/model-name
    temperature: 0.7
    streaming: true
    history:
      cut_after: 10
      summarize_after: 8
```

### 2. Добавление провайдера

В `llms.py` добавьте фабричную функцию:

```python
def create_new_llm_provider(api_key: str, model_config: dict) -> BaseChatModel:
    """Создание нового LLM провайдера."""
    # Реализация создания модели
    pass
```

### 3. Регистрация в фабрике

В `define_llm()` добавьте:

```python
elif provider == "new_provider":
    return create_new_llm_provider(api_key, model_config)
```

## Добавление нового инструмента

### 1. Создание инструмента

Создайте файл в `tools/`:

```python
from langchain_core.tools import BaseTool
from typing import Any, Dict

class NewTool(BaseTool):
    name = "new_tool"
    description = "Описание инструмента"

    def _run(self, query: str) -> str:
        # Логика инструмента
        return f"Результат для: {query}"
```

### 2. Регистрация в сервисе

В `tool_service.py` добавьте:

```python
def init_new_tool(self) -> NewTool:
    """Инициализация нового инструмента."""
    return NewTool()
```

### 3. Добавление в граф

В `graph.py` добавьте узел:

```python
new_tool_node = ModToolNode(
    tools=[tool_service.init_new_tool()],
    name="new_tool"
)
```

## Расширение API

### Добавление нового эндпоинта

В `routes.py`:

```python
@router.post("/custom")
async def custom_endpoint(request: CustomRequest) -> CustomResponse:
    """Новый эндпоинт."""
    # Логика
    return CustomResponse(result="ok")
```

### Добавление новой схемы

В `schemas.py`:

```python
class CustomRequest(BaseModel):
    param1: str
    param2: Optional[int] = None

class CustomResponse(BaseModel):
    result: str
    timestamp: datetime = Field(default_factory=datetime.now)
```

## Работа с состоянием разговора

### Структура State

```python
class State(TypedDict):
    messages: Dict[str, Messages]      # Сообщения по LLM
    user: str                         # Пользователь
    location: str                     # Локация
    additional_instructions: str      # Доп. инструкции
    llm_to_use: str                   # Текущий LLM
    last_used_llm: str               # Последний использованный
    thread_id: str                   # ID потока
    path: list                       # Путь выполнения
    summary: Dict[str, str]          # Суммаризации
```

### Редьюсеры состояния

- `add_messages_to_dict`: Добавление сообщений
- `add_path`: Управление путем выполнения
- `add_summary`: Суммаризация разговоров

## Тестирование

### Структура тестов

```
tests/
├── unit/              # Модульные тесты
├── integration/       # Интеграционные тесты
└── performance/       # Тесты производительности
```

### Запуск тестов

```bash
# Все тесты
pytest

# С покрытием
pytest --cov=src/ai_server

# Только интеграционные
pytest tests/integration/
```

### Добавление нового теста

```python
class TestNewFeature:
    def test_something(self):
        # Тест
        assert True
```

## Логирование

### Уровни логирования

```python
import logging
log = logging.getLogger(f'{APP_NAME}.module_name')

log.debug("Детальная информация")
log.info("Общая информация")
log.warning("Предупреждение")
log.error("Ошибка")
```

### Конфигурация

В `config.yaml`:

```yaml
logging:
  level: INFO
  format: "%(message)s"
  rich_tracebacks: true
```

## Производительность

### Кеширование LLM

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def get_llm_instance(model_name: str):
    return create_llm(model_name)
```

### Оптимизация запросов

- Используйте async/await
- Пул соединений для внешних API
- Кешируйте часто используемые данные

## Безопасность

### Валидация входных данных

```python
from pydantic import BaseModel, validator

class SafeRequest(BaseModel):
    user_input: str

    @validator('user_input')
    def validate_input(cls, v):
        if len(v) > 1000:
            raise ValueError('Слишком длинный ввод')
        return v
```

### Обработка секретов

```python
import os
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY не найден")
```

## Отладка

### Логирование запросов

```python
@app.middleware("http")
async def log_requests(request, call_next):
    log.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    log.info(f"Response: {response.status_code}")
    return response
```

### Профилирование

```python
import cProfile
cProfile.run('main()', 'profile.stats')
```

## Развертывание

### Development

```bash
uv run ai_server
```

### Production

```bash
# С Gunicorn
gunicorn ai_server.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Полезные команды

```bash
# Проверка типов
pyright

# Линтинг
ruff check .

# Форматирование
ruff format .

# Тесты
pytest

# Документация
# http://localhost:8000/docs
```

## Частые проблемы

### ImportError
- Проверьте PYTHONPATH
- Убедитесь в установке зависимостей

### ConfigurationError
- Проверьте config.yaml
- Проверьте переменные окружения

### LLM API Error
- Проверьте API ключи
- Проверьте лимиты использования

## Расширение функциональности

### Добавление нового типа сообщений

1. Расширить `State` в `models/state.py`
2. Добавить редьюсер в `models/state.py`
3. Обновить `ConversationService`
4. Добавить тесты

### Интеграция с новым сервисом

1. Создать клиент в `services/`
2. Добавить конфигурацию
3. Интегрировать в основной поток
4. Добавить мониторинг

## Контрибьютинг

1. Создайте ветку для фичи
2. Напишите тесты
3. Обновите документацию
4. Создайте PR

### Code Style

- Используйте type hints
- Добавляйте docstrings
- Следуйте PEP 8
- Пиши тесты
