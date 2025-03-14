# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
import datetime
from rich.traceback import install as install_rich_traceback
from kimiconfig import Config
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode 
from state import State, add_path
from llms import LLMNode, define_llm, summarize_conversation, cut_conversation
from utils import sep_line, _log_state, pretty_repr

cfg = Config()
log = logging.getLogger('ai_server.graph')
install_rich_traceback(show_locals=True)


def _update_path_in_state(state: State, path_point: str):
    """ 
    Updates path in state with given value.
    Needed for routing functions, which can't update state just by return values.
    """
    state_path = state.get('path', list(list()))
    state_path[-1].append(path_point)
    state.update({'path': state_path})


class StartNode:
    """
    Represents the starting node in the state graph.

    Methods:
        __call__(state: State): Returns the path for the start node.
    """
    def __init__(self):
        pass

    def __call__(self, state: State):
        return {'path': ['start',]}


class FinalNode:
    """
    Represents the final node in the state graph.

    Methods:
        __call__(state: State): Processes the state and returns the path for the final node.
    """
    def __init__(self):
        pass

    def __call__(self, state: State):
        # _log_state(state)
        if messages := state.get('messages', []):
            message = messages[state['llm_to_use']][-1]
        else:
            raise ValueError('No messages found in inputs')
        # last_message = state.get('messages')[-1]
        # func = cfg.runtime.user_confs.get_(state['thread_id'])['callback']  # pyright: ignore[reportAttributeAccessIssue]
        # func({'answer': message.content})

        # new_last_message = AIMessage(
        #         content=last_message.content,
        #         tool_calls=last_message.tool_calls,
        #         id=last_message.id,
        # )
        # cfg.runtime.graph.update_state(
        #     cfg.runtime.user_confs.get_lgconf(state['thread_id']), 
        #     new_last_message,
        #     as_node="final",
        # )
        # return {"messages": message.content}
        return {'path': 'final',
                'last_answer_time': datetime.datetime.now(),
                }


class ModToolNode(ToolNode):
    """
    Represents a modified tool node that processes input and invokes the tool.

    Methods:
        invoke(input, config=None, **kwargs): Invokes the tool with the given input and logs the result.
    """
    def invoke(self, input, config=None, **kwargs):
        # Replacing messages dict with current LLM's list of messages for this specific tool
        input['messages'] = input['messages'][input['llm_to_use']]
        # log.debug(f'Tool Input: {pretty_repr(input)}')
        result = super().invoke(input, config, **kwargs)
        log.debug("Tool Result:" + pretty_repr(result, max_string=200))
        # Placing result tool message back to a specific branch of llm's messages
        return {
            'messages': { input['llm_to_use']: result['messages'] }, 
            "path": "tool",
        }


def _init_graph():
    """Initializes the state graph with nodes and edges based on the configuration."""
    graph_builder = StateGraph(State)  # TODO Хорошо бы переписать это всё с фабрикой и в циклах, но боюсь потерять читабельность, надо подумать

    # from_START_to_llms_or_define_dict = {'define_llm': 'define_llm'}  # словари условной маршрутизации, наполняем в цикле ниже, потом создаем с ними условные грани
    # from_define_to_llms_or_final_dict = {'final': 'final'}

    # Стартовая нода
    graph_builder.add_node('start', StartNode())
    graph_builder.add_edge(START, 'start')

    # Нода-распределитель нейронок
    graph_builder.add_node('define_llm', define_llm)
    
    # Берем имена 2 уровня раздела models в конфиге
    # и создаем по узлу для каждой нейронки
    for llm in cfg.models.__dict__.keys():
        if llm.startswith('_'):
            continue
        graph_builder.add_node(f'{llm}_llm', LLMNode(llm))  # по ноде для каждой нейронки
        graph_builder.add_node(f'{llm}_tools', ModToolNode(tools=eval(f'cfg.runtime.tools.{llm}'),))  # по ноде тулзов для каждой
        graph_builder.add_conditional_edges(f'{llm}_llm', route_tools, {'tools': f'{llm}_tools', 'final': 'final'})  # по условной грани до тулзов или до финала
        graph_builder.add_edge(f'{llm}_tools', f'{llm}_llm')  # по обратной грани от тулзов до нейронки - без условий

        # from_START_to_llms_or_define_dict[f'{llm}_llm'] = f'{llm}_llm'  #  чтоб два раза не вставать, потребуется ниже
        # from_define_to_llms_or_final_dict[f'{llm}_llm'] = f'{llm}_llm'

    graph_builder.add_conditional_edges('start', route_llms, )  # from_START_to_llms_or_define_dict)  # грани от старта до нейронки, если она известна, иначе define_llm
    graph_builder.add_conditional_edges('define_llm', route_llms, )  # from_define_to_llms_or_final_dict)  # грани от define_llm до сетки или до финала

    # Конец
    graph_builder.add_node('final', FinalNode())
    graph_builder.add_node('summarize_conversation', summarize_conversation)
    graph_builder.add_node('cut_conversation', cut_conversation)
    graph_builder.add_conditional_edges('final', route_shorten_history)
    graph_builder.add_edge('cut_conversation', 'final')
    graph_builder.add_edge('summarize_conversation', 'final')


    graph = graph_builder.compile(
        checkpointer=cfg.runtime.memory,
        # interrupt_before=["tools"],
        # interrupt_after=["tools"])    
    )
    
    if log.getEffectiveLevel == logging.DEBUG:
        cfg.runtime.console.print(graph.get_graph().draw_ascii())
        graph.get_graph().draw_mermaid_png()
    cfg.update('runtime.graph', graph)


def route_llms(state: State) -> str:
    """Determines the next node to route to based on the current state.

    Args:
        state (State): The current state containing routing information.

    Returns:
        str: The name of the next node to route to.
    """
    sep_line('route_llms')
    _update_path_in_state(state, 'route_llms')

    if llm_to_use := state.get('llm_to_use'):
        if llm_to_use in cfg.models.__dict__.keys() and not llm_to_use.startswith('_'):
            dest = llm_to_use + '_llm'
        else:
            dest = 'define_llm'
    else:
        dest = 'define_llm'
    return dest


def route_tools(state: State) -> str:
    """Determines the next node to route to based on the tools available in the state.

    Args:
        state (State): The current state containing tool information.

    Returns:
        str: The name of the next node to route to, either 'tools' or 'final'.
    """
    sep_line('route_tools')
    _update_path_in_state(state, 'route_tools')

    ai_message: AIMessage
    if isinstance(state, list):
        ai_message = state[-1] if isinstance(state[-1], AIMessage) else AIMessage(content='')  # pyright: ignore
    elif messages := state.get('messages', []):
        ai_message = messages[state['llm_to_use']][-1]
    else:
        raise ValueError(f'No messages found in input state to tool_edge: {state}')
    if hasattr(ai_message, 'tool_calls') and len(ai_message.tool_calls) > 0:
        return 'tools'
    return 'final'


def route_shorten_history(state: State):
    """Determines whether to shorten the conversation history based on the number of messages.

    Args:
        state (State): The current state containing messages.

    Returns:
        str: The name of the next node to execute, either 'cut_conversation' or 'summarize_conversation' or 'END'.
    """
    """Return the next node to execute."""
    sep_line('route_shorten_history')
    _update_path_in_state(state, 'route_shorten_history')

    current_llm = state['llm_to_use']
    messages = state["messages"][current_llm]

    cut_history_after = getattr(cfg.models.__dict__[current_llm], "cut_history_after", 10)
    if cut_history_after > 0 and len(messages) > cut_history_after:
        return "cut_conversation"

    summarize_history_after = getattr(cfg.models.__dict__[current_llm], "summarize_history_after", 0)
    if summarize_history_after > 0 and len(messages) > summarize_history_after:
        return "summarize_conversation"

    # Otherwise we can just end
    return END

