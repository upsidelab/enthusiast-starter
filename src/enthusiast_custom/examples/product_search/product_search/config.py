from enthusiast_common.config import (
    AgentCallbackHandlerConfig,
    AgentConfigWithDefaults,
    LLMConfig,
    LLMToolConfig,
    RetrieverConfig,
    RetrieversConfig,
)
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

from .callbacks import AgentActionWebsocketCallbackHandler, ReactAgentWebsocketCallbackHandler
from .agent import ProductSearchReActAgent
from .prompt import PRODUCT_FINDER_AGENT_PROMPT
from .retrievers import ProductVectorStoreRetriever, DocumentRetriever
from .tools.product_search import ProductVectorStoreSearchTool
from .tools.product_verification import ProductVerificationTool


def get_config(conversation_id: int, streaming: bool) -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        conversation_id=conversation_id,
        prompt_template=PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"], template=PRODUCT_FINDER_AGENT_PROMPT
        ),
        agent_class=ProductSearchReActAgent,
        llm_tools=[
            LLMToolConfig(
                tool_class=ProductVectorStoreSearchTool,
            ),
            LLMToolConfig(tool_class=ProductVerificationTool),
        ],
        llm=LLMConfig(
            callbacks=[ReactAgentWebsocketCallbackHandler(conversation_id), StdOutCallbackHandler()],
            streaming=streaming,
        ),
        retrievers=RetrieversConfig(
            document=RetrieverConfig(retriever_class=DocumentRetriever),
            product=RetrieverConfig(retriever_class=ProductVectorStoreRetriever, extra_kwargs={"max_objects": 30}),
        ),
        agent_callback_handler=AgentCallbackHandlerConfig(
            handler_class=AgentActionWebsocketCallbackHandler, args={"conversation_id": conversation_id}
        ),
    )
