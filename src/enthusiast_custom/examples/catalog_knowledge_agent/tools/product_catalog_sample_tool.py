import textwrap

from enthusiast_common.injectors import BaseInjector
from enthusiast_common.tools import BaseLLMTool
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel


class ProductCatalogSampleToolInput(BaseModel):
    pass


class ProductCatalogSampleTool(BaseLLMTool):
    NAME = "product_catalog_sample"
    DESCRIPTION = "Returns a representative sample of products from the catalog. Use this first to understand what kinds of products and services are available before performing a targeted search."
    ARGS_SCHEMA = ProductCatalogSampleToolInput
    RETURN_DIRECT = False

    def __init__(
        self,
        data_set_id: int,
        llm: BaseLanguageModel,
        injector: BaseInjector,
    ):
        super().__init__(data_set_id=data_set_id, llm=llm, injector=injector)

    def run(self):
        product_retriever = self._injector.product_retriever
        sample_products = product_retriever.get_sample_products_json()
        response = f"""
            Here is a sample of products available in the catalog:
            {sample_products}
            Use the product_search tool to find products that match the user's specific query.
        """
        return textwrap.dedent(response)
