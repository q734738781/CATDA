import logging
import os
from typing import Type, List, Dict, Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for the WebSearchTool backed by Tavily."""
    query: str = Field(description="The web search query.")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum number of results to return (1-10).")
    search_depth: str = Field(default="basic", description="Search depth: 'basic' or 'advanced'.")
    days: Optional[int] = Field(default=None, description="Only include results from the last N days (optional).")
    include_answer: bool = Field(default=False, description="Include Tavily's synthesized answer.")
    include_raw_content: bool = Field(default=False, description="Include raw extracted content for each result.")
    include_domains: Optional[List[str]] = Field(default=None, description="Restrict results to these domains.")
    exclude_domains: Optional[List[str]] = Field(default=None, description="Exclude results from these domains.")


class WebSearchTool(BaseTool):
    """
    Search the web using Tavily for up-to-date information. Returns high-signal
    sources with summaries and optional synthesized answer.
    """

    name: str = "WebSearch"
    description: str = (
        "Search the web (via Tavily) for recent, high-signal information. "
        "Useful for up-to-date facts, news, and references. Supports domain filters and time windows."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        days: Optional[int] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY is not set in the environment.")
            return {"error": "Missing TAVILY_API_KEY environment variable."}

        if not query or not query.strip():
            return {"error": "Query must be a non-empty string."}

        # Normalize inputs
        max_results = max(1, min(10, int(max_results)))
        depth = (search_depth or "basic").lower()
        if depth not in ("basic", "advanced"):
            depth = "basic"

        try:
            from tavily import TavilyClient  # Lazy import to avoid hard dependency when unused
        except Exception as e:
            logger.error(f"Failed to import Tavily SDK: {e}")
            return {"error": f"Tavily SDK not installed: {e}"}

        try:
            client = TavilyClient(api_key=api_key)

            search_kwargs: Dict[str, Any] = {
                "query": query,
                "search_depth": depth,
                "max_results": max_results,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
            }
            if include_domains:
                search_kwargs["include_domains"] = include_domains
            if exclude_domains:
                search_kwargs["exclude_domains"] = exclude_domains
            if days is not None:
                try:
                    d_val = int(days)
                    if d_val > 0:
                        search_kwargs["days"] = d_val
                except Exception:
                    logger.warning("Invalid 'days' value supplied; ignoring.")

            logger.info(f"Executing Tavily search: depth={depth}, max_results={max_results}, days={days}, include_answer={include_answer}, include_raw_content={include_raw_content}")
            resp: Dict[str, Any] = client.search(**search_kwargs)

            # Ensure output is compact and JSON-serializable
            output: Dict[str, Any] = {
                "query": query,
                "results": [],
            }
            if include_answer and isinstance(resp, dict) and resp.get("answer"):
                output["answer"] = resp.get("answer")

            # Tavily returns 'results' as a list of dicts
            for item in (resp.get("results") or []):
                result_obj = {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),  # May be None unless include_raw_content=True
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                }
                output["results"].append(result_obj)

            return output

        except Exception as e:
            logger.error(f"Error during Tavily web search: {e}", exc_info=True)
            return {"error": f"Tavily search failed: {e}"}

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        days: Optional[int] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Simple async shim to sync run
        logger.warning("WebSearchTool async path not implemented; running sync version.")
        return self._run(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            days=days,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )


