# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
from datetime import datetime
from pydantic import AnyUrl
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr

from ai_server.config import cfg, APP_NAME
from ai_server.models.state import State
from ai_server.logs.utils import clean_structure, sep_line, log_diff
from ai_server.logs.themes import prompt_theme

prompt_console = Console(record=True, theme=prompt_theme)
log = logging.getLogger(f'{APP_NAME}.{__name__}')
log.addHandler(RichHandler(console=prompt_console, markup=True, log_time_format='%X'))
log.propagate = False

    
def _get_llm(
    model: str, 
    temperature: float|None = None, 
    streaming: bool=False, 
    proxy: str|None = None,
    tools: list|None = None,
    ) -> BaseChatModel|Runnable|None:

    llm: BaseChatModel|Runnable|None = None

    args = {
        'model': model,
        'streaming': streaming,
    }
    if temperature:
        args['temperature'] = temperature
    if proxy is not None:
        import httpx
        http_client = httpx.Client(proxy=cfg.proxies.__dict__.get(proxy, None))
        args['http_client'] = http_client
    # if tools:
    #     args['tools'] = tools

    if model in ('o1', 'o3-mini', 'gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview'):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(**args)
    
    elif model.lower().startswith('aihubmix_'):
        args['model'] = model[9:]
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url=cfg.llm_api.aihubmix.base_url, 
            api_key=cfg.llm_api.aihubmix.api_key, 
            **args
            )
    
    elif model.lower().startswith('openrouter_'):
        args['model'] = model[11:]
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url=cfg.llm_api.openrouter.base_url, 
            api_key=cfg.llm_api.openrouter.api_key, 
            **args
            )

    elif model.lower().startswith('deepseek'):
        model_endpoints = {
                'deepseek-r1': 'deepseek-reasoner',
                'deepseek-v3': 'deepseek-chat',
                'deepseek-chat': 'deepseek-chat',
                'deepseek-reasoner': 'deepseek-reasoner',
        }

        args['model'] = model_endpoints[model.lower().replace('_', '-')]
        args['openai_api_base'] = cfg.llm_api.deepseek.base_url
        args['api_base'] = cfg.llm_api.deepseek.base_url

        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(**args)
        # from langchain_openai.chat_models.base import BaseChatOpenAI
        # llm = BaseChatOpenAI(**args)

    # elif model.startswith('claude'):
    #     from langchain_anthropic import ChatAnthropic
    #     llm = ChatAnthropic(model=model,
    #                         temperature=temperature,
    #                         streaming=streaming,
    #                         )

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
        tool_call['id'] 
        for m in messages 
        if isinstance(m, AIMessage) and hasattr(m, 'tool_calls') 
        for tool_call in m.tool_calls
    }
    completed_calls = {
        m.tool_call_id 
        for m in messages 
        if isinstance(m, ToolMessage)
    }
    pending_calls = all_calls - completed_calls
    
    sep_line('checking tool calls status', symbol='•')
    log.debug(f'All tool calls: {pretty_repr(all_calls)}')
    log.debug(f'Completed calls: {pretty_repr(completed_calls)}')
    log.debug(f'Pending calls: {pretty_repr(pending_calls)}')
    
    return pending_calls, completed_calls, all_calls


def define_llm(state: State):
    prompt_template = ChatPromptTemplate(
        [
            ('system', cfg.prompts.define_llm),
            ('placeholder', '{message_to_define}'),
        ]
    )

    last_used_llm = state.get('last_used_llm', None)
    if last_used_llm:
        addition_to_prompt = f'Previous message was addressed to "{last_used_llm}". If you think that next message is a continuation of the previous dialogue, return the same value.',
    else:
        addition_to_prompt = ''

    message_to_define = [state['messages']['Undefined'][-1]]
    prompt = prompt_template.invoke(
        {
            'last_used_llm': addition_to_prompt,
            'message_to_define': message_to_define,
        }
    )
    answer = cfg.runtime.define_llm.invoke(prompt).content

    if answer not in cfg.agents.__dict__.keys():
        log.error(f'Defining suitable LLM went wrong: for message {message_to_define.content} was chosen: {answer}. Running "common".')
        answer = 'common'
    else:
        log.debug(f'Defined LLM: {answer}')

    return_values = {
        "llm_to_use": answer,
        "messages": { answer: message_to_define },
        "path": "define_llm",
    }

    # If answer is for all LLMs, adding it to all, who uses common
    if getattr(cfg.agents, answer).history.post_to_common:
        for name, model in vars(cfg.agents).items():
            if model.history.use_common and name != answer:
                return_values["messages"].update({name: message_to_define})

    return return_values


def cut_conversation(state: State) -> dict:
    current_llm = state['llm_to_use']
    cut_history_after = getattr(cfg.agents.__dict__[current_llm].history, "cut_after", 10)
    messages_to_cut = state['messages'][current_llm][:cut_history_after]
    
    pending_calls, _, _ = get_tool_calls_diff(messages=messages_to_cut)
    if pending_calls:
        log.debug(f'Found {len(pending_calls)} pending tool calls, can\'t cut history yet.')
        return {}
        
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_cut]
    return {
        "path": "cut_conversation",
        "messages": {current_llm: delete_messages}
    }


def summarize_conversation(state: State) -> dict:
    current_llm = state['llm_to_use']
    pending_calls, _, _ = get_tool_calls_diff(state['messages'][current_llm])
    if pending_calls:
        log.warning(f'Found {len(pending_calls)} pending tool calls, can\'t summarize yet.')
        return {}

    summary = state.get('summary', dict()).get(current_llm, "")
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
    for m in state['messages'][current_llm]:
        if isinstance(m, ToolMessage) and len(m.content) > 100:
            messages_prepared.append(ToolMessage(
                content="Skipped tool answer as too big. Assume all is ok here", 
                id=m.id, 
                tool_call_id=m.tool_call_id, 
                name=m.name
            ))
        else:
            messages_prepared.append(m)

    log.debug(f'Summarize prompt: {pretty_repr(messages_prepared, max_depth=2, max_string=100)}')
    messages_prepared.append(HumanMessage(content=summary_message))
    response = cfg.runtime.summarize_llm.invoke(messages_prepared)
    log.debug(f'Summary: {pretty_repr(response, max_depth=2)}')

    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][current_llm][:-2]]

    # Checking last 2 messages for not being part of tool call, removing if they are
    for m in state["messages"][current_llm][-2:]:
        if isinstance(m, ToolMessage) or (
            isinstance(m, AIMessage) and 
            hasattr(m, 'tool_calls') and 
            len(m.tool_calls) > 0
        ):
            delete_messages.append(RemoveMessage(id=m.id)) # pyright: ignore ReportArgumentType

    return {
        "summary": { current_llm: f'\n\n      == Previous conversation summary ==\n{response.content}' },
        "path": "summarize_conversation",
        "messages": {current_llm: delete_messages}
    }


class LLMNode:
    def __init__(self, name: str):
        self.name = name
        self.prompt_text = getattr(cfg.prompts, name, "You are smart qualified personal assistant. Answer all questions frankly.")
        # if name == 'summarize_llm':
        #     self.specific_
        self.config = getattr(cfg.agents, name)
        try:
            self.tools: list|None = getattr(cfg.runtime.tools, name)
        except:
            self.tools = None

        # Composing string from MCP resources by uris in agent config list.
        try:
            self.resources: str = '\n'.join([r.data for r in cfg.runtime.resources if str(r.metadata['uri']) in self.config.resources])
        except Exception as e:
            log.error(f'Error while compiling MCP resources for {self.name}: {e}')
            self.resources = ""

        if self.config.proxy:
            self.proxy = getattr(cfg.proxies, self.config.proxy)
        else:
            self.proxy = ""
        log.debug(f"{self.name} - {self.config.model} - {self.proxy}")

        self.llm = _get_llm(
            model=self.config.model, 
            temperature=self.config.temperature, 
            streaming=self.config.streaming, 
            proxy=self.proxy, 
            tools=self.tools
            )

        if cfg.logging.debug.llm_init:
            log.debug(self)

    async def __call__(self, state: State) -> dict:
        if not self.llm:
            log.error(f'Model "{self.name}" was not created thus not called. Skipping.')
            return { "path": self.name }
        
        # log.debug(pretty_repr(state['messages']))
        if cfg.logging.debug.messages_diff:
            log_diff(state.get('last_messages', []), state['messages'])

        prompt_template = ChatPromptTemplate(
            [
                ('system', self.prompt_text),
                ('placeholder', '{conversation}'),
            ]
        )
        prompt_substitutions = {
                'mood': cfg.runtime.mood,
                'today': datetime.now().strftime("%a, %d %b %Y, %T"),
                'username': _get_user_desc(state['user']),
                'location': _get_location_desc(state['location']),
                'summary': state.get('summary', dict()).get(self.name, ''),
                'additional_instructions': state['additional_instructions'],
                'conversation': state['messages'].get(self.name, []),
                'mcp_resources': self.resources,
            }

        prompt = prompt_template.invoke(prompt_substitutions)

        if cfg.logging.debug.prompts:
            log.debug(f'Prompt: {pretty_repr(clean_structure(prompt, ['.*metadata']), max_depth=3, max_string=10000)}')

        # log.debug(pretty_repr(clean_structure(state['messages']), max_depth=5, max_string=100))
        answer = await self.llm.ainvoke(prompt)

        return_values = {
                "messages": {self.name: answer},
                "path": self.name,
        }

        # If answer is for all LLMs, adding it to all, who uses common
        if self.config.history.post_to_common:
            for name, model in vars(cfg.agents).items():
                if model.history.use_common and name != self.name:
                    return_values["messages"].update({name: answer})

        if cfg.logging.debug.messages_diff:
            return_values["last_messages"] = state['messages']

        return return_values
    
    def __repr__(self) -> str:
        return f'''LLM Node: \n
                name: {self.name}, \n
                prompt_text: {pretty_repr(self.prompt_text, max_string=20000)}, \n
                resources: {self.resources} \n
                llm: {pretty_repr(self.llm, max_depth=3)} \n
                config: {pretty_repr(self.config)} \n
            '''
                # tools: {pretty_repr(self.tools)}. \n


def init_agents():
    define_llm = _get_llm(model=cfg.agents._define.model,
                     temperature=cfg.agents._define.temperature,
                     streaming=cfg.agents._define.streaming,
                     proxy=cfg.agents._define.proxy,
                     )
    cfg.update('runtime.define_llm', define_llm)
    summarize_llm = _get_llm(model=cfg.agents._summarize.model,
                     temperature=cfg.agents._summarize.temperature,
                     streaming=cfg.agents._summarize.streaming,
                     proxy=cfg.agents._summarize.proxy,
                     )
    cfg.update('runtime.summarize_llm', summarize_llm)
