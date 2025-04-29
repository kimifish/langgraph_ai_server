import unittest
import logging
from unittest.mock import patch, MagicMock
from src.ai_server.llms import _get_llm, define_llm, summarize_conversation, LLMNode, init_models
from ai_server.models.state import State
from ai_server.logs.themes import prompt_theme
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from kimiconfig import Config
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr

class TestLLMs(unittest.TestCase):

    def setUp(self):
        self.cfg = Config(use_dataclasses=True)
        self.prompt_console = Console(record=True, theme=prompt_theme)
        self.log = logging.getLogger('ai_server.llms')
        self.log.addHandler(RichHandler(console=self.prompt_console, markup=True, log_time_format='%X'))
        self.log.propagate = False

    def test_get_llm(self):
        # Test with valid model
        llm = _get_llm('o1', 0.7)
        self.assertIsInstance(llm, BaseChatModel)

        # Test with invalid model
        llm = _get_llm('invalid_model', 0.7)
        self.assertIsNone(llm)

    def test_define_llm(self):
        state = State(messages=[HumanMessage(content="Test message")])
        result = define_llm(state)
        self.assertIn("llm_to_use", result)
        self.assertIn("path", result)

    def test_summarize_conversation(self):
        state = State(messages=[HumanMessage(content="Test message 1"), HumanMessage(content="Test message 2")])
        result = summarize_conversation(state)
        self.assertIn("summary", result)
        self.assertIn("path", result)
        self.assertIn("messages", result)

    def test_llm_node(self):
        llm_node = LLMNode('common')
        state = State(messages=[HumanMessage(content="Test message")])
        result = llm_node(state)
        self.assertIn("messages", result)
        self.assertIn("path", result)

    def test_init_models(self):
        init_models()
        self.assertIsInstance(self.cfg.runtime.define_llm, BaseChatModel)
        self.assertIsInstance(self.cfg.runtime.summarize_llm, BaseChatModel)

if __name__ == '__main__':
    unittest.main()
