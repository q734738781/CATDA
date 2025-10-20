import logging
from typing import List, Dict, Any, Tuple

from langchain.agents import AgentExecutor
from langchain_core.callbacks.usage import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Ensure custom handlers are imported correctly
from .custom_handlers import ToolCaptureHandler

logger = logging.getLogger(__name__)

def run_agent(agent_executor: AgentExecutor, user_input: str, chat_history_tuples: List[Tuple[str | None, str | None]]) -> Dict[str, Any]:
    """
    Runs the agent executor with the given input and history, capturing tool calls and usage.

    Args:
        agent_executor: The initialized LangChain AgentExecutor.
        user_input: The user's current query.
        chat_history_tuples: The chat history from Gradio (list of tuples).

    Returns:
        A dictionary containing:
        - "answer": The agent's final response string.
        - "tools": A list of captured tool calls (query and result dicts).
        - "usage": Token usage metadata.
    """
    tool_handler = ToolCaptureHandler()
    usage_metadata = None
    ai_response = "Sorry, I encountered an issue processing your request."
    tool_records = []

    # Convert Gradio history (tuples) to LangChain Message objects
    # Ignore tool call messages (where user is None)
    langchain_history: List[BaseMessage] = []
    for user_msg, ai_msg in chat_history_tuples:
        if user_msg:
            langchain_history.append(HumanMessage(content=user_msg))
        if ai_msg:
            # Heuristic: If the message starts like a tool call, skip adding it as AI message
            # This prevents the agent from seeing its own tool calls in history.
            if not ai_msg.strip().startswith("<span style='color:#b58900'>"):
                 langchain_history.append(AIMessage(content=ai_msg))


    try:
        with get_usage_metadata_callback() as usage_cb:
            response = agent_executor.invoke(
                {"input": user_input, "chat_history": langchain_history},
                config={"callbacks": [tool_handler]}
            )
            ai_response = response.get('output', ai_response)
            usage_metadata = usage_cb.usage_metadata
            tool_records = tool_handler.records
            logger.info(f"Agent invoked. Response: {ai_response[:100]}... Tools: {len(tool_records)}, Usage: {usage_metadata}")

    except Exception as e:
        logger.error(f"Error during agent invocation: {e}", exc_info=True)
        ai_response = f"Sorry, an error occurred: {e}"
        # Attempt to get usage data even if invoke failed partially
        usage_metadata = usage_cb.usage_metadata if 'usage_cb' in locals() else None
        tool_records = tool_handler.records # Return any tools captured before the error

    return {
        "answer": ai_response,
        "tools": tool_records,
        "usage": usage_metadata
    } 