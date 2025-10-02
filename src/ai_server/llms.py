# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
from datetime import datetime
from typing import Dict, Any, cast
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr
import httpx

from ai_server.config import cfg, APP_NAME
from ai_server.models.state import State
from ai_server.logs.utils import clean_structure, sep_line, log_diff
from ai_server.logs.themes import prompt_theme

prompt_console = Console(record=True, theme=prompt_theme)
log = logging.getLogger(f"{APP_NAME}.{__name__}")
log.addHandler(RichHandler(console=prompt_console, markup=True, log_time_format="%X"))
log.propagate = False

# HTTP client pool for connection reuse
_http_client_pool: Dict[str, httpx.Client] = {}


def _get_http_client(proxy: str | None = None) -> httpx.Client:
    """
    Get or create an HTTP client for the given proxy configuration.

    This function maintains a pool of HTTP clients to reuse connections
    and improve performance for external API calls.

    Args:
        proxy: Proxy configuration string, or None for no proxy

    Returns:
        httpx.Client instance configured for the proxy
    """
    proxy_key = proxy or "no_proxy"

    if proxy_key not in _http_client_pool:
        client_kwargs = {
            "timeout": 60.0,  # 60 second timeout
            "limits": httpx.Limits(max_keepalive_connections=20, max_connections=100),
        }

        if proxy:
            proxy_url = cfg.proxies.__dict__.get(proxy, None)
            if proxy_url:
                client_kwargs["proxy"] = proxy_url

        _http_client_pool[proxy_key] = httpx.Client(**client_kwargs)
        log.debug(f"Created HTTP client for proxy: {proxy_key}")

    return _http_client_pool[proxy_key]


def cleanup_http_clients():
    """
    Clean up HTTP client pool by closing all clients.

    This should be called on application shutdown to properly close connections.
    """
    for proxy_key, client in _http_client_pool.items():
        try:
            client.close()
            log.debug(f"Closed HTTP client for proxy: {proxy_key}")
        except Exception as e:
            log.warning(f"Error closing HTTP client for {proxy_key}: {e}")

    _http_client_pool.clear()


@lru_cache(maxsize=32)
def _get_llm_cached(
    model: str,
    temperature: float | None = None,
    streaming: bool = False,
    proxy: str | None = None,
) -> BaseChatModel | Runnable | None:
    """
    Cached version of _get_llm for frequently used LLM configurations.

    This function caches LLM instances to improve performance by avoiding
    repeated initialization of the same model configurations.
    """
    return _get_llm_uncached(model, temperature, streaming, proxy, None)


def _get_llm(
    model: str,
    temperature: float | None = None,
    streaming: bool = False,
    proxy: str | None = None,
    tools: list | None = None,
) -> BaseChatModel | Runnable | None:
    """
    Get LLM instance, using cached version if no tools, uncached if tools provided.

    This is a convenience function that automatically chooses between cached
    and uncached versions based on whether tools are provided.
    """
    if tools:
        return _get_llm_uncached(model, temperature, streaming, proxy, tools)
    else:
        return _get_llm_cached(model, temperature, streaming, proxy)


def _get_llm_uncached(
    model: str,
    temperature: float | None = None,
    streaming: bool = False,
    proxy: str | None = None,
    tools: list | None = None,
) -> BaseChatModel | Runnable | None:
    llm: BaseChatModel | Runnable | None = None

    args = {
        "model": model,
        "streaming": streaming,
    }
    if temperature:
        args["temperature"] = temperature
    if proxy is not None:
        http_client = _get_http_client(proxy)
        args["http_client"] = http_client
    if tools:
        args["tools"] = tools

    if model in ("o1", "o3-mini", "gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(**args)

    elif model.lower().startswith("aihubmix_"):
        args["model"] = model[9:]
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=cfg.llm_api.aihubmix.base_url,
            api_key=cfg.llm_api.aihubmix.api_key,
            **args,
        )

    elif model.lower().startswith("openrouter_"):
        args["model"] = model[11:]
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=cfg.llm_api.openrouter.base_url,
            api_key=cfg.llm_api.openrouter.api_key,
            **args,
        )

    elif model.lower().startswith("deepseek"):
        model_endpoints = {
            "deepseek-r1": "deepseek-reasoner",
            "deepseek-v3": "deepseek-chat",
            "deepseek-chat": "deepseek-chat",
            "deepseek-reasoner": "deepseek-reasoner",
        }

        args["model"] = model_endpoints[model.lower().replace("_", "-")]
        args["openai_api_base"] = cfg.llm_api.deepseek.base_url
        args["api_base"] = cfg.llm_api.deepseek.base_url

        from langchain_deepseek import ChatDeepSeek

        llm = ChatDeepSeek(**args)

    if llm and tools:
        llm = llm.bind_tools(tools)

    return llm


def _get_location_desc(location: str) -> str:
    known_places = cfg.locations.__dict__
    if location in known_places.keys():
        return known_places[location]
    return location


def _get_user_desc(username: str) -> str:
    known_names = cfg.usernames.__dict__
    if username in known_names.keys():
        return known_names[username]
    return username


def get_tool_calls_diff(messages: list) -> tuple[set, set, set]:
    """
    Analyzes tool calls and their responses in messages.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of sets containing:
        - pending_calls: Tool calls waiting for response
        - completed_calls: Tool calls that got responses
        - all_calls: All tool call IDs
    """
    all_calls = {
        tool_call["id"]
        for m in messages
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls")
        for tool_call in m.tool_calls
    }
    completed_calls = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    pending_calls = all_calls - completed_calls

    sep_line("checking tool calls status", symbol="•")
    log.debug(f"All tool calls: {pretty_repr(all_calls)}")
    log.debug(f"Completed calls: {pretty_repr(completed_calls)}")
    log.debug(f"Pending calls: {pretty_repr(pending_calls)}")

    return pending_calls, completed_calls, all_calls


def define_llm(state: State):
    prompt_template = ChatPromptTemplate(
        [
            ("system", cfg.prompts.define_llm),
            ("placeholder", "{message_to_define}"),
        ]
    )

    last_used_llm = state.get("last_used_llm", None)
    if last_used_llm:
        addition_to_prompt = (
            f'Previous message was addressed to "{last_used_llm}". If you think that next message is a continuation of the previous dialogue, return the same value.',
        )
    else:
        addition_to_prompt = ""

    message_to_define = [state["messages"]["Undefined"][-1]]
    prompt = prompt_template.invoke(
        {
            "last_used_llm": addition_to_prompt,
            "message_to_define": message_to_define,
        }
    )
    answer = cfg.runtime.define_llm.invoke(prompt).content

    if answer not in cfg.agents.__dict__.keys():
        log.error(
            f'Defining suitable LLM went wrong: for message {message_to_define.content} was chosen: {answer}. Running "common".'
        )
        answer = "common"
    else:
        log.debug(f"Defined LLM: {answer}")

    return_values = {
        "llm_to_use": answer,
        "messages": {answer: message_to_define},
        "path": "define_llm",
    }

    # If answer is for all LLMs, adding it to all, who uses common
    answer_history = getattr(getattr(cfg.agents, answer, None), "history", None)
    if answer_history and getattr(answer_history, "post_to_common", False):
        for name, model in vars(cfg.agents).items():
            model_history = getattr(model, "history", None)
            if (
                model_history
                and getattr(model_history, "use_common", False)
                and name != answer
            ):
                return_values["messages"].update({name: message_to_define})

    return return_values


def cut_conversation(state: State) -> dict:
    current_llm = state["llm_to_use"]
    cut_history_after = getattr(
        cfg.agents.__dict__[current_llm].history, "cut_after", 10
    )
    messages_to_cut = state["messages"][current_llm][:cut_history_after]

    pending_calls, _, _ = get_tool_calls_diff(messages=messages_to_cut)
    if pending_calls:
        log.debug(
            f"Found {len(pending_calls)} pending tool calls, can't cut history yet."
        )
        return {}

    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_cut if m.id]
    return {"path": "cut_conversation", "messages": {current_llm: delete_messages}}


def summarize_conversation(state: State) -> dict:
    current_llm = state["llm_to_use"]
    pending_calls, _, _ = get_tool_calls_diff(state["messages"][current_llm])
    if pending_calls:
        log.warning(
            f"Found {len(pending_calls)} pending tool calls, can't summarize yet."
        )
        return {}

    summary = state.get("summary", dict()).get(current_llm, "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above."
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages_prepared = []
    for m in state["messages"][current_llm]:
        if isinstance(m, ToolMessage) and len(m.content) > 100:
            messages_prepared.append(
                ToolMessage(
                    content="Skipped tool answer as too big. Assume all is ok here",
                    id=m.id,
                    tool_call_id=m.tool_call_id,
                    name=m.name,
                )
            )
        else:
            messages_prepared.append(m)

    log.debug(
        f"Summarize prompt: {pretty_repr(messages_prepared, max_depth=2, max_string=100)}"
    )
    messages_prepared.append(HumanMessage(content=summary_message))
    response = cfg.runtime.summarize_llm.invoke(messages_prepared)
    log.debug(f"Summary: {pretty_repr(response, max_depth=2)}")

    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [
        RemoveMessage(id=m.id) for m in state["messages"][current_llm][:-2] if m.id
    ]

    # Checking last 2 messages for not being part of tool call, removing if they are
    for m in state["messages"][current_llm][-2:]:
        if isinstance(m, ToolMessage) or (
            isinstance(m, AIMessage)
            and hasattr(m, "tool_calls")
            and len(m.tool_calls) > 0
        ):
            if m.id:
                delete_messages.append(RemoveMessage(id=m.id))

    return {
        "summary": {
            current_llm: f"\n\n      == Previous conversation summary ==\n{response.content}"
        },
        "path": "summarize_conversation",
        "messages": {current_llm: delete_messages},
    }


class LLMNode:
    def __init__(self, name: str):
        self.name = name
        self.prompt_text = getattr(
            cfg.prompts,
            name,
            "You are smart qualified personal assistant. Answer all questions frankly.",
        )
        # if name == 'summarize_llm':
        #     self.specific_
        self.config = getattr(cfg.agents, name)
        try:
            self.tools: list | None = getattr(cfg.runtime.tools, name)
        except:
            self.tools = None

        # Composing string from MCP resources by uris in agent config list.
        try:
            resources_list = getattr(self.config, "resources", [])
            runtime_resources = getattr(cfg.runtime, "resources", [])
            self.resources: str = "\n".join(
                [
                    r.data
                    for r in runtime_resources
                    if str(r.metadata["uri"]) in resources_list
                ]
            )
        except Exception as e:
            log.error(f"Error while compiling MCP resources for {self.name}: {e}")
            self.resources = ""

        proxy_config = getattr(self.config, "proxy", None)
        if proxy_config:
            self.proxy = getattr(cfg.proxies, proxy_config)
        else:
            self.proxy = ""

        # Get model, temperature, streaming with safe defaults
        model = getattr(self.config, "model")
        temperature = getattr(self.config, "temperature", None)
        streaming = getattr(self.config, "streaming", False)

        log.debug(f"{self.name} - {model} - {self.proxy}")

        # Use cached version if no tools, uncached if tools are provided
        if self.tools:
            self.llm = _get_llm_uncached(
                model=model,
                temperature=temperature,
                streaming=streaming,
                proxy=self.proxy,
                tools=self.tools,
            )
        else:
            self.llm = _get_llm_cached(
                model=model,
                temperature=temperature,
                streaming=streaming,
                proxy=self.proxy,
            )

        if cfg.logging.debug.llm_init:
            log.debug(self)

    def prepare_prompt_template(self) -> ChatPromptTemplate:
        """
        Prepare the chat prompt template for this LLM.

        Returns:
            Configured ChatPromptTemplate instance
        """
        return ChatPromptTemplate(
            [
                ("system", self.prompt_text),
                ("placeholder", "{conversation}"),
            ]
        )

    def prepare_prompt_substitutions(self, state: State) -> Dict[str, Any]:
        """
        Prepare substitution values for the prompt template.

        Args:
            state: Current conversation state

        Returns:
            Dictionary of substitution values
        """
        # Add MCP warning for smarthome agents if MCP is not available
        additional_instructions = state["additional_instructions"]
        if self.name in ["smarthome", "smarthome_machine"] and not self.resources:
            mcp_warning = "\nВНИМАНИЕ: MCP сервер недоступен, функциональность умного дома ограничена."
            if additional_instructions:
                additional_instructions += mcp_warning
            else:
                additional_instructions = mcp_warning

        return {
            "mood": getattr(cfg.runtime, "mood", "neutral"),
            "today": datetime.now().strftime("%a, %d %b %Y, %T"),
            "username": _get_user_desc(state["user"]),
            "location": _get_location_desc(state["location"]),
            "summary": state.get("summary", {}).get(self.name, ""),
            "additional_instructions": additional_instructions,
            "conversation": state["messages"].get(self.name, []),
            "mcp_resources": self.resources,
        }

    def log_prompt_debug_info(self, prompt):
        """
        Log debug information about the prompt if enabled.

        Args:
            prompt: The prepared prompt
        """
        if cfg.logging.debug.prompts:
            log.debug(
                f'Prompt: {pretty_repr(clean_structure(prompt, [".*metadata"]), max_depth=3, max_string=10000)}'
            )

    async def execute_llm_call(self, prompt) -> Any:
        """
        Execute the LLM call with the prepared prompt.

        Args:
            prompt: The prepared prompt

        Returns:
            LLM response
        """
        return await cast(BaseChatModel, self.llm).ainvoke(prompt)

    def prepare_return_values(self, answer: Any, state: State) -> Dict[str, Any]:
        """
        Prepare the return values for the graph node.

        Args:
            answer: LLM response
            state: Current conversation state

        Returns:
            Dictionary of return values for the graph
        """
        return_values = {
            "messages": {self.name: answer},
            "path": self.name,
        }

        # If answer is for all LLMs, add it to all who use common
        history_config = getattr(self.config, "history", None)
        if history_config and getattr(history_config, "post_to_common", False):
            for name, model in vars(cfg.agents).items():
                model_history = getattr(model, "history", None)
                if (
                    model_history
                    and getattr(model_history, "use_common", False)
                    and name != self.name
                ):
                    return_values["messages"][name] = answer

        # Add last messages for debugging if enabled
        if cfg.logging.debug.messages_diff:
            return_values["last_messages"] = state["messages"]

        return return_values

    def log_message_differences(self, state: State):
        """
        Log message differences for debugging if enabled.

        Args:
            state: Current conversation state
        """
        if cfg.logging.debug.messages_diff:
            log_diff(state.get("last_messages", []), state["messages"])

    async def __call__(self, state: State) -> dict:
        """
        Execute the LLM node in the conversation graph.

        Processes the current conversation state through this LLM,
        generating a response and updating the conversation flow.

        Args:
            state: Current conversation state

        Returns:
            Updated state with LLM response and path information
        """
        if not self.llm:
            log.error(f'Model "{self.name}" was not created thus not called. Skipping.')
            return {"path": self.name}

        # Log message differences for debugging
        self.log_message_differences(state)

        # Prepare prompt
        prompt_template = self.prepare_prompt_template()
        prompt_substitutions = self.prepare_prompt_substitutions(state)
        prompt = prompt_template.invoke(prompt_substitutions)

        # Log prompt debug information
        self.log_prompt_debug_info(prompt)

        # Execute LLM call
        answer = await self.execute_llm_call(prompt)

        # Prepare return values
        return self.prepare_return_values(answer, state)

    def __repr__(self) -> str:
        return f"""LLM Node: \n
                name: {self.name}, \n
                prompt_text: {pretty_repr(self.prompt_text, max_string=20000)}, \n
                resources: {self.resources} \n
                llm: {pretty_repr(self.llm, max_depth=3)} \n
                config: {pretty_repr(self.config)} \n
            """


def init_agents():
    define_llm = _get_llm_cached(
        model=cfg.agents._define.model,
        temperature=cfg.agents._define.temperature,
        streaming=cfg.agents._define.streaming,
        proxy=getattr(cfg.agents._define, "proxy", None),
    )
    cfg.update("runtime.define_llm", define_llm)
    summarize_llm = _get_llm_cached(
        model=cfg.agents._summarize.model,
        temperature=cfg.agents._summarize.temperature,
        streaming=cfg.agents._summarize.streaming,
        proxy=getattr(cfg.agents._summarize, "proxy", None),
    )
    cfg.update("runtime.summarize_llm", summarize_llm)
