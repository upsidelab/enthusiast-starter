from enthusiast_common.config import (
    AgentCallbackHandlerConfig,
    AgentConfigWithDefaults,
    LLMConfig,
    LLMToolConfig,
)
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

from .callbacks import AgentActionWebsocketCallbackHandler, ReactAgentWebsocketCallbackHandler
from .agent import DataExtractionReActAgent
from .prompt import DATA_EXTRACTION_AGENT_PROMPT
from .tools.data_extraction import DataExtractionTool
from .tools.data_verification import DataVerificationTool


def get_config(conversation_id: int, streaming: bool) -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        conversation_id=conversation_id,
        prompt_template=PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"], template=DATA_EXTRACTION_AGENT_PROMPT
        ),
        agent_class=DataExtractionReActAgent,
        llm_tools=[
            LLMToolConfig(
                tool_class=DataVerificationTool,
            ),
            LLMToolConfig(
                tool_class=DataExtractionTool,
            )
        ],
        llm=LLMConfig(
            callbacks=[ReactAgentWebsocketCallbackHandler(conversation_id), StdOutCallbackHandler()],
            streaming=streaming,
        ),
        agent_callback_handler=AgentCallbackHandlerConfig(
            handler_class=AgentActionWebsocketCallbackHandler, args={"conversation_id": conversation_id}
        ),
    )
