from enthusiast_common.agents import BaseAgentConfigProvider, ConfigType
from enthusiast_common.config import AgentConfigWithDefaults

from .agent import ExampleDocumentContextAgent
from .prompt import DOCUMENT_CONTEXT_AGENT_SYSTEM_PROMPT


class ExampleDocumentContextAgentConfigProvider(BaseAgentConfigProvider):
    def get_config(self, config_type: ConfigType = ConfigType.CONVERSATION) -> AgentConfigWithDefaults:
        return AgentConfigWithDefaults(
            system_prompt=DOCUMENT_CONTEXT_AGENT_SYSTEM_PROMPT,
            agent_class=ExampleDocumentContextAgent,
            tools=ExampleDocumentContextAgent.TOOLS,
        )
