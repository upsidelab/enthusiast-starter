import json

from enthusiast_common.injectors import BaseInjector
from enthusiast_common.tools import BaseLLMTool
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field


class ProductSearchToolInput(BaseModel):
    query: str = Field(description="A natural-language description of the product or service to search for.")


class ProductSearchTool(BaseLLMTool):
    NAME = "product_search"
    DESCRIPTION = "Searches the product catalog using a natural-language description and returns matching products. Use this to find specific products or services that meet the user's criteria."
    ARGS_SCHEMA = ProductSearchToolInput
    RETURN_DIRECT = False

    def __init__(
        self,
        data_set_id: int,
        llm: BaseLanguageModel,
        injector: BaseInjector,
    ):
        super().__init__(data_set_id=data_set_id, llm=llm, injector=injector)

    def run(self, query: str) -> str:
        product_retriever = self._injector.product_retriever
        relevant_products = product_retriever.find_products_matching_query(query)
        if not relevant_products:
            return "No products found matching the query. Try rephrasing or broadening the search."

        serialized = product_retriever.product_details_as_json(relevant_products)
        return json.dumps(serialized)
