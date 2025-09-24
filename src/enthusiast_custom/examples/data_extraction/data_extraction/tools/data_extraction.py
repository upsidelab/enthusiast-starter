from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from enthusiast_common.injectors import BaseInjector
from enthusiast_common.tools import BaseLLMTool
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field



class DataExtractionToolInput(BaseModel):
    url: str = Field(description="URL for website to get data from")


class DataExtractionTool(BaseLLMTool):
    NAME = "data_extraction_tool"
    DESCRIPTION = (
        "Always use this tool. Use this tool to data from web page."
    )
    ARGS_SCHEMA = DataExtractionToolInput
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


    def _is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])

    def run(self, url: str) -> str:
        if not self._is_valid_url(url):
            return "Invalid URL"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
