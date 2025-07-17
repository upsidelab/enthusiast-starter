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
```
3. To enable new plugin, add it to settings_override.py:
```python
CATALOG_DOCUMENT_SOURCE_PLUGINS = {
    "PDF Plugin": "<path_to_file>.PDFDocumentSourcePlugin"
}
```
Now this custom plugin will be available in Document source section.
`config` variable used in above example could be provided while adding any source plugin, in this example it was used to provide urls to documents.

### Next, let's move to agent. 
1. Create a directory for you agent (e.g. `pdf_agent`). Then inside it create `agent.py` file:
```python
from enthusiast_common.agents import BaseAgent
from enthusiast_common.injectors import BaseInjector
from enthusiast_common.tools.base import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate


class ExamplePDFAgent(BaseAgent):
    def __init__(
        self,
        tools: list[BaseTool],
        llm: BaseLanguageModel,
        prompt: ChatPromptTemplate,
        conversation_id: int,
        injector: BaseInjector,
        callback_handler: BaseCallbackHandler | None = None,
    ):
        super().__init__(
            tools=tools,
            llm=llm,
            prompt=prompt,
            conversation_id=conversation_id,
            callback_handler=callback_handler,
            injector=injector,
        )
        self._agent_executor = self._create_agent_executor()

    def _create_agent_executor(self, **kwargs) -> AgentExecutor:
        tools = self._create_tools()
        agent = create_tool_calling_agent(self._llm, tools, self._prompt)
        return AgentExecutor(
            agent=agent, tools=tools, verbose=True, memory=self._injector.chat_summary_memory, **kwargs
        )

    def _create_tools(self):
        return [tool_class.as_tool() for tool_class in self._tools]

    def get_answer(self, input_text: str) -> str:
        agent_output = self._agent_executor.invoke(
            {"input": input_text}, config={"callbacks": [self._callback_handler] if self._callback_handler else []}
        )
        return agent_output["output"]
```
2. Create Prompt in prompt.py file.
```python
PDF_AGENT_SYSTEM_PROMPT="""
You are a helpful agent, answering questions about pdf document. Always use context tool
"""
```

3. Create context retrieving tool:
```python
import tiktoken
from enthusiast_common.injectors import BaseInjector
from enthusiast_common.tools import BaseLLMTool
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field


class ContextSearchToolInput(BaseModel):
    full_user_request: str = Field(description="user's full request")


class ContextSearchTool(BaseLLMTool):
    NAME = "context_search_tool"
    DESCRIPTION = "Use it to get context from pdf required for answering questions"
    ARGS_SCHEMA = ContextSearchToolInput
    RETURN_DIRECT = False

    def __init__(
        self,
        data_set_id: int,
        llm: BaseLanguageModel,
        injector: BaseInjector,
    ):
        super().__init__(data_set_id=data_set_id, llm=llm, injector=injector)
        self.data_set_id = data_set_id
        self.llm = llm
        self.injector = injector

    def run(self, full_user_request: str):
        document_retriever = self.injector.document_retriever
        relevant_documents = document_retriever.find_content_matching_query(full_user_request)
        content  = [document.content for document in relevant_documents]

        return content
```

4. Create configuration inside `config.py` file:
```python
from enthusiast_common.config import AgentConfigWithDefaults, LLMConfig, LLMToolConfig
from langchain_core.prompts import ChatPromptTemplate

from .tools.pdf_context_tool import ContextSearchTool
from .agent import ExamplePDFAgent
from .prompt import PDF_AGENT_SYSTEM_PROMPT


def get_config(conversation_id: int, streaming: bool) -> AgentConfigWithDefaults:
    return AgentConfigWithDefaults(
        conversation_id=conversation_id,
        prompt_template=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    PDF_AGENT_SYSTEM_PROMPT,
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ),
        agent_class=ExamplePDFAgent,
        llm_tools=[
            LLMToolConfig(
                tool_class=ContextSearchTool,
            )
        ],
    )
```
5. Finally add your agent to `settings_override.py`:
```python
AVAILABLE_AGENTS = {
    "PDF Agent": "enthusiast_custom.pdf_agent",
}

```
Now Agent is available in UI to chat with it.