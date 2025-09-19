from enthusiast_common.config import (
    AgentCallbackHandlerConfig,
    AgentConfigWithDefaults,
    LLMConfig,
)
from enthusiast_common.config.base import PromptTemplateConfig, CallbackHandlerConfig
from langchain_core.callbacks import StdOutCallbackHandler

from .callbacks import AgentActionWebsocketCallbackHandler, ReactAgentWebsocketCallbackHandler
from .agent import DataExtractionReActAgent
from .prompt import DATA_EXTRACTION_AGENT_PROMPT


def get_config() -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        prompt_template=PromptTemplateConfig(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"], template=DATA_EXTRACTION_AGENT_PROMPT
        ),
        agent_class=DataExtractionReActAgent,
        tools=DataExtractionReActAgent.TOOLS,
        llm=LLMConfig(
            callbacks=[
                CallbackHandlerConfig(handler_class=ReactAgentWebsocketCallbackHandler),
                CallbackHandlerConfig(handler_class=StdOutCallbackHandler),
            ],
        ),
        agent_callback_handler=AgentCallbackHandlerConfig(
            handler_class=AgentActionWebsocketCallbackHandler
        ),
    )
