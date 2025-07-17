from pathlib import Path

import requests
from enthusiast_common import DocumentSourcePlugin, DocumentDetails
from langchain_community.document_loaders import PyPDFLoader


class PDFDocumentSourcePlugin(DocumentSourcePlugin):
    def __init__(self, data_set_id: int, config: dict):
        super().__init__(data_set_id, config)

    def fetch(self) -> list[DocumentDetails]:
        title = "A practical guide to building agents"
        url = "https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf"

        results = []
        try:
            response = requests.get(url)
            response.raise_for_status()

            temp_path = Path(f"/tmp/temp.pdf")
            with open(temp_path, "wb") as f:
                f.write(response.content)

            loader = PyPDFLoader(str(temp_path))
            for index, page in enumerate(loader.lazy_load()):
                results.append(DocumentDetails(url=f"{url}/{index}", title=title, content=page.page_content))
            print(len(results))
            return results

        except Exception as e:
            print(f"Failed to load {url} ({title}): {e}")
