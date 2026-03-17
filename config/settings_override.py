# Put your custom settings in this file
# They will override the settings defined in Enthusiast's pecl/settings.py

# Register new eCommerce integrations here
CATALOG_ECOMMERCE_INTEGRATION_PLUGINS = ["enthusiast_source_medusa.MedusaIntegration"]

# Register new product sources here
CATALOG_PRODUCT_SOURCE_PLUGINS = [
    "enthusiast_source_sample.SampleProductSource",
    "enthusiast_source_medusa.MedusaProductSource",
]

# Register new document sources here
CATALOG_DOCUMENT_SOURCE_PLUGINS = [
    "enthusiast_source_sample.SampleDocumentSource",
    "enthusiast_custom.examples.pdf_documents_plugin.PDFDocumentSourcePlugin",
]

# Register new agents here
AVAILABLE_AGENTS = [
    'enthusiast_agent_catalog_enrichment.CatalogEnrichmentAgent',
    'enthusiast_agent_order_intake.OrderIntakeAgent',
    'enthusiast_agent_user_manual_search.UserManualSearchAgent',
    'enthusiast_agent_product_search.ProductSearchAgent',
    'enthusiast_custom.examples.document_context_agent.ExampleDocumentContextAgent'
]
