import logging
from pathlib import Path

import requests
from enthusiast_common import DocumentSourcePlugin, DocumentDetails
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

class PDFDocumentSourcePlugin(DocumentSourcePlugin):
    def __init__(self, data_set_id: int, config: dict):
        super().__init__(data_set_id, config)

    def fetch(self) -> list[DocumentDetails]:
        results = []
        data = self.config.get("data", [])

        for document in data:
            url = document.get("url")
            title = document.get("title")
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
