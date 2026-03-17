from enthusiast_common.config import AgentConfigWithDefaults
from enthusiast_common.config.prompts import ChatPromptTemplateConfig, Message, MessageRole

from .agent import ExampleDocumentContextAgent
from .prompt import DOCUMENT_CONTEXT_AGENT_SYSTEM_PROMPT


def get_config() -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        prompt_template=ChatPromptTemplateConfig(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=DOCUMENT_CONTEXT_AGENT_SYSTEM_PROMPT,
                ),
                Message(role=MessageRole.PLACEHOLDER, content="{chat_history}"),
                Message(role=MessageRole.USER, content="{input}"),
                Message(role=MessageRole.PLACEHOLDER, content="{agent_scratchpad}"),
            ]
        ),
        agent_class=ExampleDocumentContextAgent,
        tools=ExampleDocumentContextAgent.TOOLS,
    )