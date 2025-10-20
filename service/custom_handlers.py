from langchain_core.callbacks.base import BaseCallbackHandler
from typing import List, Dict, Any, Optional

class ToolCaptureHandler(BaseCallbackHandler):
    """Callback handler to capture tool execution details."""
    def __init__(self):
        self.records: List[Dict[str, Any]] = []  # List to store {'query': ..., 'result': ...}

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        inputs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool starts."""
        # Capture any tool call; store tool name
        tool_name = serialized.get("name", "UnknownTool")
        if tool_name:
            # Sometimes input_str might be a dict string like '{{\"cypher_query\": \"MATCH...\"}}'
            # Try to parse it if needed, otherwise use as is.
            query_to_store = input_str
            try:
                # Langchain might pass the input as a dictionary string
                import json
                input_dict = json.loads(input_str)
                # If dict, store pretty json else leave plaintext
                if isinstance(input_dict, dict):
                    query_to_store = json.dumps(input_dict, indent=2)
            except (json.JSONDecodeError, TypeError):
                # If it's not a JSON dict string, assume it's the query directly
                pass

            self.records.append({"tool": tool_name, "query": query_to_store, "result": ""})
            print(f"Tool Start Captured ({tool_name}): {query_to_store}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool ends."""
        if self.records and self.records[-1]["result"] == "":
            self.records[-1]["result"] = output
            print(f"Tool End Captured: {self.records[-1]['result']}") # Debug print

    # Ignore other callback methods like on_llm_start, on_chat_model_start etc.
    # Add them if detailed tracing is needed later.

    # Ensure compatibility with potential sync/async contexts if needed
    # by defining async counterparts if the environment requires it,
    # though for this simple capture, sync methods are usually sufficient. 