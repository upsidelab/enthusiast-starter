from enthusiast_common.agents import BaseAgent
from enthusiast_common.config.base import LLMToolConfig
from enthusiast_common.injectors import BaseInjector
from enthusiast_common.tools.base import BaseTool
from enthusiast_custom.examples.pdf_agent.pdf_agent.tools.pdf_context_tool import ContextSearchTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate


class ExamplePDFAgent(BaseAgent):
    TOOLS = [LLMToolConfig(tool_class=ContextSearchTool)]
    def __init__(
        self,
        tools: list[BaseTool],
        llm: BaseLanguageModel,
        prompt: ChatPromptTemplate,
        conversation_id: int,
        injector: BaseInjector,
        callback_handler: BaseCallbackHandler | None = None,
    ):
        super().__init__(
            tools=tools,
            llm=llm,
            prompt=prompt,
            conversation_id=conversation_id,
            callback_handler=callback_handler,
            injector=injector,
        )
        self._agent_executor = self._create_agent_executor()

    def _create_agent_executor(self, **kwargs) -> AgentExecutor:
        tools = self._create_tools()
        agent = create_tool_calling_agent(self._llm, tools, self._prompt)
        return AgentExecutor(
            agent=agent, tools=tools, verbose=True, memory=self._injector.chat_summary_memory, **kwargs
        )

    def _create_tools(self):
        return [tool_class.as_tool() for tool_class in self._tools]

    def get_answer(self, input_text: str) -> str:
        agent_output = self._agent_executor.invoke(
            {"input": input_text}, config={"callbacks": [self._callback_handler] if self._callback_handler else []}
        )
        return agent_output["output"]