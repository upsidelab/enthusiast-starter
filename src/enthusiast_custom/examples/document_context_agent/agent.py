from enthusiast_agent_tool_calling import BaseToolCallingAgent
from enthusiast_common.config.base import LLMToolConfig

from .tools import ContextSearchTool


class ExampleDocumentContextAgent(BaseToolCallingAgent):
    AGENT_KEY = "enthusiast-agent-example-document-context"
    NAME = "Example Document Context Agent"
    
    TOOLS = [LLMToolConfig(tool_class=ContextSearchTool)]