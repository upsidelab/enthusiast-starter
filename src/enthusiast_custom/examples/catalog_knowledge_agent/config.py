from enthusiast_common.agents import BaseAgentConfigProvider, ConfigType
from enthusiast_common.config import AgentConfigWithDefaults

from .agent import CatalogKnowledgeAgent
from .prompt import CATALOG_KNOWLEDGE_AGENT_SYSTEM_PROMPT


class CatalogKnowledgeConfigProvider(BaseAgentConfigProvider):
    def get_config(self, config_type: ConfigType = ConfigType.CONVERSATION) -> AgentConfigWithDefaults:
        return AgentConfigWithDefaults(
            system_prompt=CATALOG_KNOWLEDGE_AGENT_SYSTEM_PROMPT,
            agent_class=CatalogKnowledgeAgent,
            tools=CatalogKnowledgeAgent.TOOLS,
        )
