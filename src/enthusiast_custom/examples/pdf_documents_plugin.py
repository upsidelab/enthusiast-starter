import logging
from pathlib import Path

import requests
from enthusiast_common import DocumentSourcePlugin, DocumentDetails
from enthusiast_common.utils import RequiredFieldsModel
from langchain_community.document_loaders import PyPDFLoader
from pydantic import Field, Json

logger = logging.getLogger(__name__)


class PDFSourceConfig(RequiredFieldsModel):
    data: Json = Field(title="PDF urls", description="Json containing values where key is document name and value is url.", default='{"filename.pdf": "https://example.com"}')

class PDFDocumentSourcePlugin(DocumentSourcePlugin):
    CONFIGURATION_ARGS = PDFSourceConfig

    def fetch(self) -> list[DocumentDetails]:
        results = []
        data = self.CONFIGURATION_ARGS.data
        for title, url in data.items():
            try:
                response = requests.get(url)
                response.raise_for_status()

                temp_path = Path(f"/tmp/temp.pdf")
                with open(temp_path, "wb") as f:
                    f.write(response.content)

                loader = PyPDFLoader(str(temp_path))
                for index, page in enumerate(loader.lazy_load()):
                    results.append(DocumentDetails(url=f"{url}/{index}", title=title, content=page.page_content))
                return results

            except Exception as e:
                logger.error(f"Failed to load {url} ({title}): {e}")
