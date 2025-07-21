from enthusiast_agent_re_act import BaseReActAgent
from enthusiast_common.config import LLMToolConfig
from enthusiast_common.utils import RequiredFieldsModel
from enthusiast_custom.examples.product_search.product_search.tools.product_search import ProductVectorStoreSearchTool
from enthusiast_custom.examples.product_search.product_search.tools.product_verification import ProductVerificationTool
from pydantic import Field


class ProductSearchReActAgentInput(RequiredFieldsModel):
    products_type: str = Field(title="Products type", description="Type of product to search for")


class ProductSearchReActAgent(BaseReActAgent):
    PROMPT_INPUT = ProductSearchReActAgentInput
    TOOLS = [LLMToolConfig(tool_class=ProductVectorStoreSearchTool), LLMToolConfig(tool_class=ProductVerificationTool)]


    def get_answer(self, input_text: str) -> str:
        agent_executor = self._build_agent_executor()
        agent_output = agent_executor.invoke(
            {"input": input_text, "products_type": self.PROMPT_INPUT.products_type},
            config=self._build_invoke_config(),
        )
        return agent_output["output"]
