<div align="center">
  <a href="https://upsidelab.io/tools/enthusiast" />
    <img src="https://github.com/user-attachments/assets/966204c3-ff69-47b2-a247-9f9cfa4e5b7d" height="150px" alt="Enthusiast">
  </a>
</div>

<h1 align="center">enthusiast.</h1>

<p align="center">Your open-souce AI agent for e-commerce.</p>
<div align="center">
  <strong>
    <a href="https://upsidelab.io/tools/enthusiast/docs/getting-started/installation">Get Started</a> |
    <a href="https://upsidelab.io/tools/enthusiast/docs">View Docs</a> |
    <a href="https://upsidelab.io/tools/enthusiast">Website</a>
  </strong>
</div>

## Introduction
Enthusiast is your open-source AI agent for e-commerce. Connect it to your product catalog, add content that describes your products and brand, and scale your team’s capabilities.

## Starter Pack

This repository provides everything you need to build custom agents and integrations, and deploy them to production with ease.

### Building a custom agent

Use the `src/` directory for your code—create agents or plugins using the interfaces defined in `enthusiast-common`, and enable them via `settings_override.py`.

The included Dockerfile installs Enthusiast, builds your custom package, and configures the system to use it.

To bootstrap your environment variables for Docker, run:
```shell
cp config/env.sample config/env
``` 
to bootstrap environment variables for the docker setup.

Then start the application locally:
```shell
docker compose build && docker compose up
```


## Example: Creating a PDF Documents Agent

This guide walks you through creating a simple **Documents Plugin** that allows you to insert PDF documents into a database and create embeddings.  
Then we’ll create a custom **Agent** that can answer questions based on those PDFs.

### Firstly create a simple documents plugin - which will allow you to insert pdf documents into database and create embeddings.
1. Add needed dependencies:

``` bash
poetry add langchain-community pypdf
```

2. Next, create a new file and add Plugin class
```python
import logging
from pathlib import Path

import requests
from enthusiast_common import DocumentSourcePlugin, DocumentDetails
from enthusiast_common.utils import RequiredFieldsModel
from langchain_community.document_loaders import PyPDFLoader
from pydantic import Field

logger = logging.getLogger(__name__)


class PDFSourceConfig(RequiredFieldsModel):
    url: str = Field(title="URL", description="URL to PDF file.")
    filename: str = Field(title="Filename", description="PDF filename.")


class PDFDocumentSourcePlugin(DocumentSourcePlugin):
    CONFIGURATION_ARGS = PDFSourceConfig
    NAME = "PDF Document Source"

    def fetch(self) -> list[DocumentDetails]:
        results = []
        title = self.CONFIGURATION_ARGS.filename
        url = self.CONFIGURATION_ARGS.url
        try:
            response = requests.get(url)
            response.raise_for_status()

            temp_path = Path("/tmp/temp.pdf")
            with open(temp_path, "wb") as f:
                f.write(response.content)

            loader = PyPDFLoader(str(temp_path))
            for index, page in enumerate(loader.lazy_load()):
                results.append(DocumentDetails(url=f"{url}/{index}", title=title, content=page.page_content))
            return results

        except Exception as e:
            logger.error(f"Failed to load {url} ({title}): {e}")
```
3. To enable new plugin, add it to settings_override.py:
```python
CATALOG_DOCUMENT_SOURCE_PLUGINS = [
    "enthusiast_custom.examples.pdf_documents_plugin.PDFDocumentSourcePlugin",
]
```

Now this custom plugin will be available in Document source section. When adding a source, provide the `url` (HTTP URL to the PDF) and `filename` fields.

### Next, let's move to agent. 

1. Create the agent directory structure:

```
src/enthusiast_custom/
    __init__.py
    examples/
        document_context_agent/
            __init__.py
            agent.py
            config.py
            prompt.py
            tools/
                __init__.py
                document_context_tool.py
```

2. Create `agent.py`:

```python
from enthusiast_agent_tool_calling import BaseToolCallingAgent
from enthusiast_common.config.base import LLMToolConfig

from .tools import ContextSearchTool


class ExampleDocumentContextAgent(BaseToolCallingAgent):
    AGENT_KEY = "enthusiast-agent-example-document-context"
    NAME = "Example Document Context Agent"

    TOOLS = [LLMToolConfig(tool_class=ContextSearchTool)]
```

3. Create `prompt.py`:

```python
DOCUMENT_CONTEXT_AGENT_SYSTEM_PROMPT = """
You are a helpful agent, answering questions about documents and resources mentioned in them.

Whenever the user asks a question, always use the document context tool first to extract
relevant context before answering.
"""
```

4. Create `tools/document_context_tool.py`:

```python
from enthusiast_common.injectors import BaseInjector
from enthusiast_common.tools import BaseLLMTool
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field


class ContextSearchToolInput(BaseModel):
    full_user_request: str = Field(description="user's full request")


class ContextSearchTool(BaseLLMTool):
    NAME = "context_search_tool"
    DESCRIPTION = "Use it to get context from documents required for answering questions"
    ARGS_SCHEMA = ContextSearchToolInput
    RETURN_DIRECT = False

    def __init__(self, data_set_id: int, llm: BaseLanguageModel, injector: BaseInjector):
        super().__init__(data_set_id=data_set_id, llm=llm, injector=injector)

    def run(self, full_user_request: str):
        relevant_documents = self.injector.document_retriever.find_content_matching_query(full_user_request)
        return [doc.content for doc in relevant_documents]
```

5. Create `config.py`:

```python
from enthusiast_common.agents import BaseAgentConfigProvider, ConfigType
from enthusiast_common.config import AgentConfigWithDefaults

from .agent import ExampleDocumentContextAgent
from .prompt import DOCUMENT_CONTEXT_AGENT_SYSTEM_PROMPT


class ExampleDocumentContextAgentConfigProvider(BaseAgentConfigProvider):
    def get_config(self, config_type: ConfigType = ConfigType.CONVERSATION) -> AgentConfigWithDefaults:
        return AgentConfigWithDefaults(
            system_prompt=DOCUMENT_CONTEXT_AGENT_SYSTEM_PROMPT,
            agent_class=ExampleDocumentContextAgent,
            tools=ExampleDocumentContextAgent.TOOLS,
        )
```

6. Export from `src/enthusiast_custom/__init__.py`:

```python
from .examples.document_context_agent import ExampleDocumentContextAgent, ExampleDocumentContextAgentConfigProvider

__all__ = ["ExampleDocumentContextAgent", "ExampleDocumentContextAgentConfigProvider"]
```

7. Register the agent in `config/settings_override.py`:

```python
AVAILABLE_AGENTS = [
    "enthusiast_custom.ExampleDocumentContextAgent",
]
```

> **Note:** The agent must be registered via the top-level `enthusiast_custom` module path (not the nested submodule path). The AgentRegistry resolves config providers by scanning the module at the registered path — nested paths cause lookup failures.

Now Agent is available in UI to chat with it.