from enthusiast_agent_tool_calling import BaseToolCallingAgent
from enthusiast_common.config.base import LLMToolConfig

from .tools import DocumentRetrievalTool, ProductCatalogSampleTool, ProductSearchTool


class CatalogKnowledgeAgent(BaseToolCallingAgent):
    AGENT_KEY = "enthusiast-agent-catalog-knowledge"
    NAME = "Catalog Knowledge Agent"
    TOOLS = [
        LLMToolConfig(tool_class=ProductCatalogSampleTool),
        LLMToolConfig(tool_class=ProductSearchTool),
        LLMToolConfig(tool_class=DocumentRetrievalTool),
    ]
