from enthusiast_agent_tool_calling import BaseToolCallingAgent
from enthusiast_common.config.base import LLMToolConfig

from .tools import ContextSearchTool


class ExamplePDFAgent(BaseToolCallingAgent):
    AGENT_KEY = "enthusiast-agent-pdf-agent"
    NAME = "PDF agent"
    
    TOOLS = [LLMToolConfig(tool_class=ContextSearchTool)]