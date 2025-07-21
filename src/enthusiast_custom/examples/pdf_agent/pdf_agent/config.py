from enthusiast_common.config import AgentConfigWithDefaults
from enthusiast_common.config.base import ChatPromptTemplateConfig

from .agent import ExamplePDFAgent
from .prompt import PDF_AGENT_SYSTEM_PROMPT


def get_config() -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        chat_prompt_template=ChatPromptTemplateConfig(messages=
            [
                (
                    "system",
                    PDF_AGENT_SYSTEM_PROMPT
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ),
        agent_class=ExamplePDFAgent,
        tools=ExamplePDFAgent.TOOLS
    )