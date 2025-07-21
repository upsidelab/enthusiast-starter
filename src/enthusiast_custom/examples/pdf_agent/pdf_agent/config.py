from enthusiast_common.config import AgentConfigWithDefaults, LLMConfig, LLMToolConfig
from langchain_core.prompts import ChatPromptTemplate

from .tools.pdf_context_tool import ContextSearchTool
from .agent import ExamplePDFAgent
from .prompt import PDF_AGENT_SYSTEM_PROMPT


def get_config(conversation_id: int, streaming: bool) -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        conversation_id=conversation_id,
        prompt_template=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    PDF_AGENT_SYSTEM_PROMPT,
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ),
        agent_class=ExamplePDFAgent,
        llm_tools=[
            LLMToolConfig(
                tool_class=ContextSearchTool,
            )
        ],
    )