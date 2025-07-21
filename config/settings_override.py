# Put your custom settings in this file
# They will override the settings defined in Enthusiast's pecl/settings.py

# Register new product sources here
CATALOG_PRODUCT_SOURCE_PLUGINS = {
    "Sample Product Source": "enthusiast_source_sample.SampleProductSource",
}

# Register new document sources here
CATALOG_DOCUMENT_SOURCE_PLUGINS = {
    "Sample Document Source": "enthusiast_source_sample.SampleDocumentSource",
    "Fetch PDFs": "enthusiast_custom.examples.pdf_documents_plugin.PDFDocumentSourcePlugin",
}

AVAILABLE_AGENTS: dict[str, dict[str, str]] = {
    "product_search_agent": {
        "name": "Product Search Agent",
        "agent_directory_path": "enthusiast_custom.examples.product_search.product_search",
    },
    "pdf_agent": {
        "name": "PDF Agent",
        "agent_directory_path": "enthusiast_custom.examples.pdf_agent.pdf_agent",
    },
    "question_answer_agent": {
        "name": "Question Answer Agent",
        "agent_directory_path": "agent.core.agents.tool_calling_agent",
    },
}

DEFAULT_AGENT: dict = {
    "type": "question_answer_agent",
    "name": "Default Agent",
    "description": "Default agent",
    "config": {"tools": [{}], "agent_args": {}, "prompt_input": {}, "prompt_extension": {}},
}
