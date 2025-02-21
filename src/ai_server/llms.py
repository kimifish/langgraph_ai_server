# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
from datetime import datetime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from kimiconfig import Config
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.pretty import pretty_repr
from state import State
from utils import prompt_theme, sep_line

cfg = Config(use_dataclasses=True)
prompt_console = Console(record=True, theme=prompt_theme)
log = logging.getLogger('ai_server.llms')
log.addHandler(RichHandler(console=prompt_console, markup=True, log_time_format='%X'))
log.propagate = False
install_rich_traceback(show_locals=True)

    
def _get_llm(model: str, temperature: float|None = None, streaming: bool=False, tools: list|None = None) -> BaseChatModel|Runnable|None:
    llm: BaseChatModel|Runnable|None = None

    args = {
        'model': model,
        'streaming': streaming,
    }
    if temperature:
        args['temperature'] = temperature
    if tools:
        args['tools'] = tools

    if model in ('o1', 'o3-mini', 'gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview'):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(**args)

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


def all_tool_calls_got_answers(messages: list) -> bool:
    ai_message_ids = {tool_call['id'] for m in messages if isinstance(m, AIMessage) and hasattr(m, 'tool_calls') for tool_call in m.tool_calls}
    tool_message_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    sep_line('checking tool answers', symbol='•')
    log.debug(pretty_repr(ai_message_ids))
    log.debug(pretty_repr(tool_message_ids))
    return ai_message_ids == tool_message_ids


def define_llm(state: State):
    log.debug("define_llm")

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

    prompt = prompt_template.invoke(
        {
            'last_used_llm': addition_to_prompt,
            'message_to_define': [state['messages']['Undefined'][-1]],
        }
    )
    answer = cfg.runtime.define_llm.invoke(prompt).content

    if answer not in cfg.models.__dict__.keys():
        log.error(f'Defining suitable LLM went wrong: for message {state["messages"]['Undefined'][-1].content} was chosen: {answer}. Running "common".')
        answer = 'common'
    else:
        log.debug(f'Defined LLM: {answer}')

    return {
        "llm_to_use": answer,
        "messages": { answer: [state['messages']['Undefined'][-1]] },
        "path": "define_llm",
    }


def cut_conversation(state: State) -> dict:
    current_llm = state['llm_to_use']
    cut_history_after = getattr(cfg.models.__dict__[current_llm], "cut_history_after", 10)
    messages_to_cut = state['messages'][current_llm][:cut_history_after]
    if not all_tool_calls_got_answers(messages=messages_to_cut):
        log.debug('Some tool calls isn\'t completed, can\'t cut history yet.')
        return {}
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_cut]
    return {
        "path": "cut_conversation",
        "messages": {current_llm: delete_messages}
    }


def summarize_conversation(state: State) -> dict:
    current_llm = state['llm_to_use']
    if not all_tool_calls_got_answers(state['messages'][current_llm]):
        log.debug('Some tool calls isn\'t completed, can\'t summarize yet.')
        return {}

    summary = state.get('summary', dict()).get(current_llm, "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages_prepared = []
    for m in state['messages'][current_llm]:
        if isinstance(m, ToolMessage) and len(m.content) > 100:
            messages_prepared.append(ToolMessage(content="Skipped tool answer as too big. Assume all is ok here", id = m.id, tool_call_id = m.tool_call_id, name = m.name ))
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
        if isinstance(m, ToolMessage) or ( isinstance(m, AIMessage) and hasattr(m, 'tool_calls') and len(m.tool_calls) > 0):
            delete_messages.append(RemoveMessage(id=m.id))

    return {
        "summary": { current_llm: f'\n\n      == Previous conversation summary ==\n{response.content}' },
        "path": "summarize_conversation",
        "messages": {current_llm: delete_messages}
    }


class LLMNode:
    def __init__(self, name: str):
        self.name = name
        self.prompt_text = eval(f'cfg.prompts.{name}')
        llm_config = eval(f'cfg.models.{name}')
        try:
            self.tools: list|None = eval(f'cfg.runtime.tools.{name}')
        except:
            self.tools = None
        self.temperature = getattr(llm_config, 'temperature', None)
        self.streaming = getattr(llm_config, 'streaming', False)
        self.llm = _get_llm(llm_config.model, self.temperature, self.streaming, self.tools)
        log.debug(self)

    def __call__(self, state: State) -> dict:
        log.debug(f'{self.name}_llm')

        if not self.llm:
            log.error(f'Model "{self.name}" was not created thus not called. Skipping.')
            return { "path": self.name }

        prompt_template = ChatPromptTemplate(
            [
                ('system', self.prompt_text),
                ('placeholder', '{conversation}'),
            ]
        )
        # messages = summarize_prev_history(state['messages'])
        prompt = prompt_template.invoke(
            {
                'mood': cfg.runtime.mood,
                'today': datetime.now().strftime("%a, %d %b %Y, %T"),
                'username': _get_user_desc(state['user']),
                'location': _get_location_desc(state['location']),
                'summary': state.get('summary', dict()).get(self.name, ''),
                'additional_instructions': state['additional_instructions'],
                'conversation': state['messages'][self.name]  #[1:],
            }
        )
        log.debug(f'Prompt: {pretty_repr(prompt, max_depth=4, max_string=100)}')
        answer = self.llm.invoke(prompt)
        return {
                "messages": {self.name: answer},
                "path": self.name,
        }
    
    def __repr__(self) -> str:
        return f'''LLM Node: \n
                name: {self.name}, \n
                llm: {pretty_repr(self.llm)}, \n
                tools: {pretty_repr(self.tools)}. \n
            '''


def _init_models():
    define_llm = ChatOpenAI(model=cfg.models._define.model,
                     temperature=cfg.models._define.temperature,
                     streaming=cfg.models._define.streaming,
                     )
    cfg.update('runtime.define_llm', define_llm)
    summarize_llm = ChatOpenAI(model=cfg.models._summarize.model,
                     temperature=cfg.models._summarize.temperature,
                     streaming=cfg.models._summarize.streaming,
                     )
    cfg.update('runtime.summarize_llm', summarize_llm)
