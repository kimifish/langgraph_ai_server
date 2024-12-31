import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.system import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from kimiconfig import Config
from state import State

cfg = Config(use_dataclasses=True)
log = logging.getLogger('ai_server.llms')

    
def _get_llm(model: str, temperature: float, streaming: bool=False, tools: list|None = None) -> BaseChatModel|Runnable|None:
    llm: BaseChatModel|Runnable|None = None
    if model in ('o1', 'gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview'):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model,
                         temperature=temperature,
                         streaming=streaming,
                         )
    # elif model.startswith('claude'):
    #     from langchain_anthropic import ChatAnthropic
    #     llm = ChatAnthropic(model=model,
    #                         temperature=temperature,
    #                         streaming=streaming,
    #                         )

    if llm and tools:
         llm = llm.bind_tools(tools)

    return llm


def define_llm(state: State):
    log.debug("define_llm")

    prompt = SystemMessage(
        content=cfg.prompts.define_llm,
    )
    
    payload = [prompt, state['messages'][-1], ]
    answer = cfg.runtime.define_llm.invoke(payload).content

    if answer not in ['common', 'smarthome', 'shell_assistant', 'code_assistant', 'school_tutor']:
        log.error(f'Defining suitable LLM went wrong: for message {state["messages"][-1].content} was chosen: {answer}. Running "common".')
        answer = 'common'
    else:
        log.debug(f'Defined LLM: {answer}')

    return {
        # "messages": [cfg.runtime.common_llm_with_search.invoke(state['messages'])],  # pyright: ignore[reportAttributeAccessIssue]
        "llm_to_use": answer,
        "path": "define_llm",
    }


class LLMNode:
    def __init__(self, name: str):
        # self.prompt_text = eval(f'cfg.prompts.{name}')
        # llm_config = eval(f'cfg.models.{name}')
        # try:
        #     self.tools: list|None = eval(f'cfg.runtime.tools.{name}')
        # except:
        #     self.tools = None
        self.name = name
        self.prompt_text = getattr(cfg.prompts, name, None)
        llm_config = getattr(cfg.models, name, None)
        self.tools = getattr(cfg.runtime.tools, name, None)
        self.llm = _get_llm(llm_config.model, llm_config.temperature, llm_config.streaming, self.tools)
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
        prompt = prompt_template.invoke(
            {
                'mood': cfg.runtime.mood,
                'username': state['user'],
                'location': state['location'],
                'additional_instructions': state['additional_instructions'],
                'conversation': state['messages'][1:],
            }
        )
        log.debug(f'{prompt=}')
        answer = self.llm.invoke(prompt)
        return {
            "messages": answer,
            "path": self.name,
        }
    
    def __repr__(self) -> str:
        return f'''LLM Node: \n
                name: {self.name}, \n
                llm: {self.llm}, \n
                tools: {self.tools}. \n
            '''


def _init_models():
    define_llm = ChatOpenAI(model=cfg.models.common.model,  # pyright: ignore[reportAttributeAccessIssue]
                     temperature=0,
                     streaming=False,
                     )
    cfg.update('runtime.define_llm', define_llm)

