import logging
from kimiconfig import Config
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode 
from state import State
from llms import LLMNode, define_llm
from utils import _log_state

cfg = Config()
log = logging.getLogger('ai_server.graph')


class FinalNode:
    """ На всякий случай ещё однa нодa в конце, вдруг пригодится. """

    def __init__(self):
        pass

    def __call__(self, state: State):
        _log_state(state)
        if messages := state.get('messages', []):
            message = messages[-1]
        else:
            raise ValueError('No messages found in inputs')
        last_message = state.get('messages')[-1]
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
        return {'path': 'final'}


def _init_graph():
    graph_builder = StateGraph(State)  # TODO Хорошо бы переписать это всё с фабрикой и в циклах, но боюсь потерять читабельность, надо подумать

    from_START_to_llms_or_define_dict = {'define_llm': 'define_llm'}  # словари условной маршрутизации, наполняем в цикле ниже, потом создаем с ними условные грани
    from_define_to_llms_or_final_dict = {'final': 'final'}

    # Нода-распределитель нейронок
    graph_builder.add_node('define_llm', define_llm)
    
    # Берем имена 2 уровня раздела models в конфиге
    # и создаем по узлу для каждой нейронки
    for llm in cfg.models.__dict__.keys():
        graph_builder.add_node(f'{llm}_llm', LLMNode(llm))  # по ноде для каждой нейронки
        graph_builder.add_node(f'{llm}_tools', ToolNode(eval(f'cfg.runtime.tools.{llm}')))  # по ноде тулзов для каждой
        graph_builder.add_conditional_edges(f'{llm}_llm', route_tools, {'tools': f'{llm}_tools', 'final': 'final'})  # по условной грани до тулзов или до финала
        graph_builder.add_edge(f'{llm}_tools', f'{llm}_llm')  # по обратной грани от тулзов до нейронки - без условий

        from_START_to_llms_or_define_dict[f'{llm}_llm'] = f'{llm}_llm'  #  чтоб два раза не вставать, потребуется ниже
        from_define_to_llms_or_final_dict[f'{llm}_llm'] = f'{llm}_llm'

    graph_builder.add_conditional_edges(START, route_llms, from_START_to_llms_or_define_dict)  # грани от старта до нейронки, если она известна, иначе define_llm
    graph_builder.add_conditional_edges('define_llm', route_llms, from_define_to_llms_or_final_dict)  # грани от define_llm до сетки или до финала

    # Конец
    graph_builder.add_node('final', FinalNode())
    graph_builder.add_edge('final', END)

    graph = graph_builder.compile(
        checkpointer=cfg.runtime.memory,  # pyright: ignore[reportAttributeAccessIssue]
        # interrupt_before=["tools"],
        # interrupt_after=["tools"])    
    )
    
    if log.getEffectiveLevel == logging.DEBUG:
        cfg.runtime.console.print(graph.get_graph().draw_ascii())
        # graph.get_graph().draw_mermaid_png()
    cfg.update('runtime.graph', graph)


def route_llms(state: State):
    state_path = state.get('path', [])
    state_path.append('route_llms')
    state.update({'path': state_path})

    if llm_to_use := state.get('llm_to_use'):
        match llm_to_use:
            case 'smarthome':
                dest = 'smarthome_llm'
            case 'shell_assistant':
                dest = 'shell_assistant_llm'
            case 'code_assistant':
                dest = 'code_assistant_llm'
            case 'school_tutor':
                dest = 'school_tutor_llm'
            case 'common':
                dest = 'common_llm'
            case _:
                dest = 'define_llm'
    else:
        dest = 'define_llm'
    log.debug(f'route_llms → {dest}')
    return dest


def route_tools(state: State):
    state_path = state.get('path', [])
    state_path.append('route_tools')
    state.update({'path': state_path})
    ai_message: AIMessage
    if isinstance(state, list):
        ai_message = state[-1] if isinstance(state[-1], AIMessage) else AIMessage(content='')
    elif messages := state.get('messages', []):
        ai_message = messages[-1]
    else:
        raise ValueError(f'No messages found in input state to tool_edge: {state}')
    if hasattr(ai_message, 'tool_calls') and len(ai_message.tool_calls) > 0:
        log.debug(f'route_tools → tools')
        return 'tools'
    log.debug(f'route_tools → final')
    return 'final'

