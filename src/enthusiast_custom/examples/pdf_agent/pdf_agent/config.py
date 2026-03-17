from enthusiast_common.config import AgentConfigWithDefaults
from enthusiast_common.config.prompts import ChatPromptTemplateConfig, Message, MessageRole

from .agent import ExamplePDFAgent
from .prompt import PDF_AGENT_SYSTEM_PROMPT


def get_config() -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        prompt_template=ChatPromptTemplateConfig(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=PDF_AGENT_SYSTEM_PROMPT,
                ),
                Message(role=MessageRole.PLACEHOLDER, content="{chat_history}"),
                Message(role=MessageRole.USER, content="{input}"),
                Message(role=MessageRole.PLACEHOLDER, content="{agent_scratchpad}"),
            ]
        ),
        agent_class=ExamplePDFAgent,
        tools=ExamplePDFAgent.TOOLS,
    )