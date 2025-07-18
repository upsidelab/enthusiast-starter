# Put your custom settings in this file
# They will override the settings defined in Enthusiast's pecl/settings.py

CATALOG_PRODUCT_SOURCE_PLUGINS = {
    "Sample Product Source": "enthusiast_source_sample.SampleProductSource",
}

CATALOG_DOCUMENT_SOURCE_PLUGINS = {
    "Sample Document Source": "enthusiast_source_sample.SampleDocumentSource",
    "Fetch PDFs": "enthusiast_custom.examples.pdf_documents_plugin.PDFDocumentSourcePlugin",
}

AVAILABLE_AGENTS = {
    "Product Search Agent": "agent.core.agents.product_search_react_agent",
    "Question Answer Agent": "agent.core.agents.tool_calling_agent",
    "PDF Query Agent": "enthusiast_custom.examples.pdf_agent",
}
