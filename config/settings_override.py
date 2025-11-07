# Put your custom settings in this file
# They will override the settings defined in Enthusiast's pecl/settings.py

# Register new product sources here
CATALOG_PRODUCT_SOURCE_PLUGINS = {
    "Sample Product Source": "enthusiast_source_sample.SampleProductSource",
    "Medusa": "enthusiast_source_medusa.MedusaProductSource",
}

# Register new document sources here
CATALOG_DOCUMENT_SOURCE_PLUGINS = {
    "Sample Document Source": "enthusiast_source_sample.SampleDocumentSource",
    "Fetch PDFs": "enthusiast_custom.examples.pdf_documents_plugin.PDFDocumentSourcePlugin",
}

AVAILABLE_AGENTS: dict[str, dict[str, str]] = {
    "ocr": {
        "name": "OCR To Order Agent",
        "agent_directory_path": "enthusiast_agent_ocr_to_order",
    },
    "product_search": {
        "name": "Product Search Agent",
        "agent_directory_path": "enthusiast_agent_product_search",
    },
    "catalog_enrichment": {
        "name": "Catalog Enrichment Agent",
        "agent_directory_path": "enthusiast_agent_catalog_enrichment",
    },
    "user_manual": {
        "name": "User Manual Agent",
        "agent_directory_path": "enthusiast_agent_user_manual_search",
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
