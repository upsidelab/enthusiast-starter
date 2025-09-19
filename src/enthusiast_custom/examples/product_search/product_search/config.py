from enthusiast_common.config.base import (
    AgentCallbackHandlerConfig,
    AgentConfigWithDefaults,
    LLMConfig,
    RetrieverConfig,
    RetrieversConfig,
)
from enthusiast_common.config.base import PromptTemplateConfig, CallbackHandlerConfig
from langchain_core.callbacks import StdOutCallbackHandler

from .callbacks import AgentActionWebsocketCallbackHandler, ReactAgentWebsocketCallbackHandler
from .agent import ProductSearchReActAgent
from .prompt import PRODUCT_FINDER_AGENT_PROMPT
from .retrievers import ProductVectorStoreRetriever, DocumentRetriever


def get_config() -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        prompt_template=PromptTemplateConfig(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"], template=PRODUCT_FINDER_AGENT_PROMPT
        ),
        agent_class=ProductSearchReActAgent,
        tools=ProductSearchReActAgent.TOOLS,
        llm=LLMConfig(
            callbacks=[
                CallbackHandlerConfig(handler_class=ReactAgentWebsocketCallbackHandler),
                CallbackHandlerConfig(handler_class=StdOutCallbackHandler),
            ],
        ),
        retrievers=RetrieversConfig(
            document=RetrieverConfig(retriever_class=DocumentRetriever),
            product=RetrieverConfig(retriever_class=ProductVectorStoreRetriever, extra_kwargs={"max_objects": 30}),
        ),
        agent_callback_handler=AgentCallbackHandlerConfig(handler_class=AgentActionWebsocketCallbackHandler),
    )
