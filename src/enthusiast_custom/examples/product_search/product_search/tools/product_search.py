from typing import Any

from enthusiast_common.injectors import BaseInjector
from enthusiast_common.tools import BaseLLMTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ProductVectorStoreSearchInput(BaseModel):
    full_user_request: str = Field(description="user's full request")
    keyword: str = Field(
        description="one-word keyword which will determine an attribute of product for postgres search. It can be color, country, shape"
    )


class ProductVectorStoreSearchTool(BaseLLMTool):
    NAME = "search_matching_products"
    DESCRIPTION = (
        "It's tool for vector store search use it with suitable phrases when you need to find matching products"
    )
    ARGS_SCHEMA = ProductVectorStoreSearchInput
    RETURN_DIRECT = False

    def __init__(
        self,
        data_set_id: int,
        llm: BaseLanguageModel,
        injector: BaseInjector | None,
    ):
        super().__init__(data_set_id=data_set_id, llm=llm, injector=injector)
        self.data_set_id = data_set_id
        self.llm = llm
        self.injector = injector

    def run(self, full_user_request: str, keyword: str) -> list[Any]:
        product_retriever = self.injector.product_retriever
        relevant_products = product_retriever.find_content_matching_query(full_user_request, keyword)
        context = [product.content for product in relevant_products]
        return context
