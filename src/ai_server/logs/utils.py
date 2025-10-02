# pyright: basic
# pyright: reportAttributeAccessIssue=false

from typing import Any, List
import logging
import re
from rich.pretty import pretty_repr
from deepdiff.diff import DeepDiff
from pydantic import BaseModel, ConfigDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from ai_server.models.state import State


log = logging.getLogger(f"{__name__}")


def _log_state(state: State, tabs: int = 0):
    log.debug(pretty_repr(state, indent_size=4, max_string=50))


def _parse_key(key: str):
    # Regular expression pattern to match root['path1']['path2']...[index]
    pattern = r"^root(\['(?:[^']+)'\])+\[(\d+)\]$"
    path_pattern = r"'\s*([^']*)\s*'"  # Pattern to extract individual path segments

    main_match = re.match(pattern, key)
    if main_match:
        # Extract all path segments
        path_segments = re.findall(path_pattern, main_match.group(0))

        # Extract the index, assuming the last part of the main match
        index = int(main_match.group(2))
        return path_segments, index
    else:
        return [], 0


def sep_line(
    name: str, length: int = 90, color: str = "light steel blue", symbol: str = "─"
):
    line_length = (length - len(name) - 2) // 2
    line = f"{symbol * line_length} {name} {symbol * line_length}"
    while len(line) < length:
        line += symbol
    while len(line) > length:
        line = line[:-1]
    log.debug(f"[{color}]{line}[/]")


def log_diff(dict1, dict2):
    diff = DeepDiff(dict1, dict2)
    for key, value in diff.items():
        try:
            if key.endswith("added"):
                for k, v in value.items():
                    log.debug(f"[chartreuse2 underline] + {k}:[/]\n{pretty_repr(v)}")
            elif key.endswith("removed"):
                for k, v in value.items():
                    log.debug(f"[deep_pink4 underline] - {k}:[/]\n{pretty_repr(v)}")
            elif key.endswith("changed"):
                for k, v in value.items():
                    log.debug(f"[orange3 underline] ~ {k}:[/]\n{pretty_repr(v)}")
            else:
                log.debug(f"{key}:" + pretty_repr(value))
        except AttributeError:
            log.error(f"Key {key} does not contain items.")


class HumanMessage_(BaseModel):
    model_config = ConfigDict(extra="allow")


class AIMessage_(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolMessage_(BaseModel):
    model_config = ConfigDict(extra="allow")


types = {
    HumanMessage: HumanMessage_,
    AIMessage: AIMessage_,
    ToolMessage: ToolMessage_,
}
# Предполагаем, что BaseMessage имеет структуру, похожую на словарь,
# или поля доступны по именам. Если это не так, потребуется уточнение.
# Для примера, будем считать, что можно итерироваться по items() как у словаря.
# Если BaseMessage - это класс из библиотеки вроде langchain, возможно,
# потребуется импоортировать его и проверить его структуру.
# Пока используем Any для гибкости.


def clean_structure(
    data: Any,
    regex_patterns: List[str] = ["metadata", "additional_kwargs", "id", "^example$"],
) -> Any:
    """
    Рекурсивно обходит структуру данных (списки, словари, объекты BaseMessage)
    и удаляет ветви, ключи которых соответствуют заданным регулярным выражениям.

    Args:
        data: Входная структура данных.
        regex_patterns: Список строк с регулярными выражениями для удаления ключей/полей.

    Returns:
        Очищенная структура данных.
    """
    if isinstance(data, dict):
        cleaned_dict = {}
        for key, value in data.items():
            # Проверяем ключ на соответствие любому из регулярных выражений
            if not any(re.search(pattern, str(key)) for pattern in regex_patterns):
                cleaned_dict[key] = clean_structure(value, regex_patterns)
        return cleaned_dict
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            cleaned_list.append(clean_structure(item, regex_patterns))
        return cleaned_list
    # Добавляем обработку для объектов, которые являются BaseModel или имеют метод model_dump/dict
    elif (
        isinstance(data, BaseModel)
        or hasattr(data, "model_dump")
        or hasattr(data, "dict")
    ):
        cleaned_obj_data = {}
        # Пытаемся получить данные объекта как словарь (предпочитаем model_dump для Pydantic v2+)
        try:
            obj_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        except Exception:
            # Если не удалось получить словарь, используем vars() как запасной вариант
            obj_data = vars(data) if hasattr(data, "__dict__") else {}

        for key, value in obj_data.items():
            # Проверяем ключ/имя атрибута на соответствие любому из регулярных выражений
            if not any(re.search(pattern, str(key)) for pattern in regex_patterns):
                cleaned_obj_data[key] = clean_structure(value, regex_patterns)

        # Создаем новый объект того же типа с очищенными данными
        # Используем словарь types для маппинга оригинальных классов на заглушки
        original_type = type(data)
        if original_type in types:
            try:
                # Создаем пустой экземпляр класса-заглушки
                new_instance = types[original_type]()
                # Вручную присваиваем очищенные атрибуты
                for key, value in cleaned_obj_data.items():
                    setattr(new_instance, key, value)
                return new_instance
            except Exception as e:
                # Если не удалось создать объект или присвоить атрибуты
                log.error(
                    f"Не удалось создать или заполнить экземпляр заглушки для {original_type}: {e}"
                )
                # Возвращаем очищенный словарь как запасной вариант
                return cleaned_obj_data
        else:
            # Если тип объекта не найден в словаре types, возвращаем очищенный словарь
            return cleaned_obj_data

    # Добавляем обработку для других объектов с __dict__, которые не являются BaseModel
    elif hasattr(data, "__dict__"):
        cleaned_obj_data = {}
        obj_data = vars(data)

        for key, value in obj_data.items():
            if not any(re.search(pattern, str(key)) for pattern in regex_patterns):
                cleaned_obj_data[key] = clean_structure(value, regex_patterns)
        # Для других объектов без специфичной заглушки, возвращаем очищенный словарь
        return cleaned_obj_data

    else:
        # Возвращаем неизмененные примитивные типы и объекты без __dict__
        return data
