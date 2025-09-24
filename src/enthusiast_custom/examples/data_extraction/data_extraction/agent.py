import json

from enthusiast_agent_re_act import BaseReActAgent
from enthusiast_common.config import LLMToolConfig
from enthusiast_common.utils import RequiredFieldsModel
from enthusiast_custom.examples.data_extraction.data_extraction.tools.data_extraction import DataExtractionTool
from pydantic import Field

from enthusiast_custom.examples.data_extraction.data_extraction.tools.data_verification import DataVerificationTool


class DataExtractionAgentInput(RequiredFieldsModel):
    products_type: str = Field(title="Products type", description="Type of product to search for")
    output_format: str = Field(title="Output format", description="Output format for product search")


class DataExtractionReActAgent(BaseReActAgent):
    AGENT_ARGS = DataExtractionAgentInput
    TOOLS = [LLMToolConfig(tool_class=DataExtractionTool), LLMToolConfig(tool_class=DataVerificationTool)]
    def get_answer(self, input_text: str) -> str:
        agent_executor = self._build_agent_executor()
        agent_output = agent_executor.invoke(
            {
                "input": input_text,
                "output_format": self.AGENT_ARGS.output_format,
                "products_type": self.AGENT_ARGS.products_type,
            },
            config=self._build_invoke_config()
        )
        result = agent_output["output"]
        try:
            parsed = json.loads(result)
            result = json.dumps(parsed, indent=4)
        except json.JSONDecodeError:
            pass

        return result
