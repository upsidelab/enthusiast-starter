# Example Document Context Agent

An agent that answers questions about documents and resources mentioned in them. It uses a context search tool to retrieve relevant content from ingested documents before responding, rather than asking the user for details.

## Activation

Add the agent to `AVAILABLE_AGENTS` in `config/settings_override.py`:

```python
AVAILABLE_AGENTS = [
    # ... other agents ...
    'enthusiast_custom.examples.document_context_agent.ExampleDocumentContextAgent',
]
```

## Implementation

### Agent (`agent.py`)

`ExampleDocumentContextAgent` extends `BaseToolCallingAgent` and is identified by the key `enthusiast-agent-example-document-context`. It is configured with a single tool: `ContextSearchTool`.

### Tool (`tools/document_context_tool.py`)

`ContextSearchTool` takes the user's full request and uses the injected `document_retriever` to find matching content across ingested documents. It returns a list of content chunks relevant to the query.

### Prompt (`prompt.py`)

The system prompt instructs the agent to always call the context tool first — whether the user is asking about a document directly or about a resource/topic mentioned in one — before formulating an answer.

### Config (`config.py`)

`get_config()` returns an `AgentConfigWithDefaults` with a standard chat prompt template (system message, chat history, user input, agent scratchpad) wired to `ExampleDocumentContextAgent` and its tools.
