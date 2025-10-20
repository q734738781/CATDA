import gradio as gr
import logging
import sys
import os
import json
from typing import List, Tuple

# Ensure project root is discoverable for imports if needed
# This structure assumes launch_gradio.py handles path adjustments if run from root
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# Project imports - Need setup_agent and run_agent
# Using relative imports now
try:
    from ..service.agent_setup import setup_agent
    from ..service.agent_wrapper import run_agent
except ImportError as e:
    print(f"Error importing service modules in gradio_app: {e}")
    # If running this file directly, relative imports will fail.
    # It needs to be run via launch_gradio.py (preferably with python -m)
    raise

logger = logging.getLogger(__name__)

# Global variable to hold the agent executor instance
# Initialized by launch_ui function
agent_executor_instance = None

def build_ui(agent_executor):
    """Builds the Gradio UI components."""
    with gr.Blocks(theme='ParityError/Interstellar', css="./style.css") as demo: # Load css relative to this file
        gr.Markdown("# Catalysis Expert Agent")

        chatbot = gr.Chatbot(
            label="Chat History",
            bubble_full_width=False,
            avatar_images=(None, "https://img.icons8.com/external-kiranshastry-gradient-kiranshastry/64/external-robot-artificial-intelligence-kiranshastry-gradient-kiranshastry.png"), # Optional: User/AI avatars
            type='messages' # Use the recommended message format
        )
        token_box = gr.Markdown("**Last Query Token Usage:** Waiting for first query...") # Changed label
        cumulative_token_box = gr.Markdown("**Session Total Token Usage:** Input: 0 | Output: 0 | Total: 0") # New box for cumulative
        state_hist = gr.State([]) # Stores List[Tuple[str | None, str | None]]
        cumulative_tokens_state = gr.State({'total_input': 0, 'total_output': 0}) # New state for cumulative tokens

        with gr.Row():
            in_price = gr.Textbox(label="Input token price per 1M ($)", value="0.0", scale=1) # Default 0.0
            out_price = gr.Textbox(label="Output token price per 1M ($)", value="0.0", scale=1) # Default 0.0
            # Removed Apply Prices button as it wasn't effective without more complex state handling
            # apply_price_btn = gr.Button("Apply Prices", variant="secondary", scale=1)

        cost_box = gr.Markdown("**Last Query Estimated Cost:** $0.000000") # Changed label
        cumulative_cost_box = gr.Markdown("**Session Total Estimated Cost:** $0.000000") # New box for cumulative cost

        def user_submit(
            message: str,
            history: List[dict],
            current_in_price: str,
            current_out_price: str,
            current_cumulative_tokens: dict # Added state input
        ):
            """Handles user message submission, calls agent, updates history and UI."""
            # Default token text in case of early exit or error
            new_token_text = "**Last Query Token Usage:** Error processing request"
            new_cost_text = cost_box.value # Keep previous cost if error
            # Initialize cumulative texts - they will be updated if successful
            new_cumulative_token_text = cumulative_token_box.value
            new_cumulative_cost_text = cumulative_cost_box.value
            updated_cumulative_tokens = current_cumulative_tokens # Use current state by default

            if not message:
                # Return empty input, unchanged history, and default/unchanged texts
                return "", history, new_token_text, new_cost_text, new_cumulative_token_text, new_cumulative_cost_text, updated_cumulative_tokens

            logger.info(f"User input: {message}")
            # Convert history from list of dicts (messages format) to list of tuples for run_agent
            history_for_agent = []
            for msg_data in history:
                # Ensure content is not None before adding to history tuples
                user_content = msg_data.get("content") if msg_data.get("role") == "user" else None
                assistant_content = msg_data.get("content") if msg_data.get("role") == "assistant" else None

                # Only add if at least one part has content (filters potential empty messages)
                if user_content is not None or assistant_content is not None:
                     # Check for tool call formatting (markdown, etc.) and skip if it's assistant message
                     # This prevents sending the formatted tool outputs back to the agent
                     is_tool_output = False
                     if assistant_content and assistant_content.strip().startswith("<hr>"):
                         is_tool_output = True
                     elif assistant_content and "🛠️ Tool:" in assistant_content:
                         is_tool_output = True

                     if not is_tool_output:
                        history_for_agent.append((user_content, assistant_content))
            logger.debug(f"History for agent (tuples): {history_for_agent}")


            # Ensure the executor is initialized
            if agent_executor is None:
                 logger.error("Agent executor not initialized!")
                 # Append error in messages format
                 history.append({'role': 'assistant', 'content': "Error: Agent backend is not ready."})
                 # Return empty input, updated history, and error token/cost text
                 return "", history, new_token_text, "**Last Query Estimated Cost:** Agent Error", new_cumulative_token_text, new_cumulative_cost_text, updated_cumulative_tokens # Updated return

            # Call the agent service function with the converted history
            result = run_agent(agent_executor, message, history_for_agent)

            # 1. Append user message in messages format
            history.append({'role': 'user', 'content': message})

            # 2. Append tool calls/results (if any) as a single system message
            # Display tool calls regardless of agent success
            if result.get("tools"): # Use .get for safety
                 logger.info(f"Formatting {len(result['tools'])} tool calls for display.")
                 all_tools_md_parts = []
                 for i, rec in enumerate(result["tools"]):
                     # Format tool call using Markdown for code blocks and spans for color
                     # Ensure newlines around code fences for better rendering
                     tool_name = rec.get('tool', 'UnknownTool')
                     tool_query = rec.get('query', 'No query captured')
                     tool_result = rec.get('result', 'No result captured')
                     
                     # Attempt to parse and re-format JSON results for better display
                     formatted_tool_result = tool_result # Default to original
                     if isinstance(tool_result, str):
                         try:
                             parsed_json = json.loads(tool_result)
                             # Re-dump with ensure_ascii=False and indentation
                             formatted_tool_result = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                         except json.JSONDecodeError:
                             # If it's not valid JSON, just use the original string
                             logger.debug(f"Tool result for {tool_name} is not valid JSON, displaying as is.")
                             # formatted_tool_result already holds the original string
                         except TypeError:
                             # Handle cases where tool_result might not be string-like (though rec.get should handle this)
                             logger.warning(f"Tool result for {tool_name} was not a string or JSON decodable, displaying as is.")

                     tool_md_part = (
                         f"<hr>\n"
                         f"<span style='color:#b58900; font-weight:bold;'>🛠️ Tool: {tool_name}  (Call {i+1}/{len(result['tools'])})</span>\n\n"
                         f"**Input:**\n```\n{tool_query}\n```\n\n" # Changed label
                         f"<span style='color:#268bd2; font-weight:bold;'>Output:</span>\n\n" # Changed label
                         f"```json\n{formatted_tool_result}\n```\n" # Use potentially reformatted result
                     )
                     all_tools_md_parts.append(tool_md_part)

                 # Combine all parts into a single markdown string
                 combined_tools_md = "\n".join(all_tools_md_parts)
                 # Append combined tools output as assistant message
                 history.append({'role': 'assistant', 'content': combined_tools_md})
                 # Add a separator after the tool calls block (also as assistant message for simplicity)
                 # history.append({'role': 'assistant', 'content': "<hr>"}) # Removed redundant separator

            # 3. Append final assistant answer (if available)
            assistant_msg = result.get("answer", "Agent did not return an answer.")
            history.append({'role': 'assistant', 'content': assistant_msg})


            # 4. Prepare token usage display text & cost (for last query and cumulative)
            if result.get("usage"): # Use .get for safety
                try:
                    # Find the first usage block (assuming one model for now)
                    usage_key = list(result["usage"].keys())[0]
                    usage_data = result["usage"][usage_key]
                    in_tokens = usage_data.get('input_tokens', 0)
                    out_tokens = usage_data.get('output_tokens', 0)
                    total_tokens = usage_data.get('total_tokens', 0)

                    # Update last query text
                    new_token_text = (
                        f"**Last Query Token Usage ({usage_key.capitalize()}) – Total:** {total_tokens}"
                        f" | **Input:** {in_tokens}"
                        f" | **Output:** {out_tokens}"
                    )
                    logger.info(f"Last query token usage updated: {new_token_text}")

                    # Update cumulative tokens state
                    updated_cumulative_tokens = current_cumulative_tokens.copy() # Avoid modifying the input dict directly before return
                    updated_cumulative_tokens['total_input'] += in_tokens
                    updated_cumulative_tokens['total_output'] += out_tokens
                    total_cumulative = updated_cumulative_tokens['total_input'] + updated_cumulative_tokens['total_output']

                    # Update cumulative token text
                    new_cumulative_token_text = (
                        f"**Session Total Token Usage:** Input: {updated_cumulative_tokens['total_input']}"
                        f" | Output: {updated_cumulative_tokens['total_output']}"
                        f" | Total: {total_cumulative}"
                    )
                    logger.info(f"Cumulative token usage updated: {new_cumulative_token_text}")


                    # Cost calc using current values from the textboxes
                    try:
                        in_price_val = float(current_in_price) # Use passed value
                        out_price_val = float(current_out_price) # Use passed value

                        # Calculate last query cost
                        cost = (in_tokens * in_price_val + out_tokens * out_price_val) / 1_000_000
                        new_cost_text = f"**Last Query Estimated Cost:** ${cost:.6f}"

                        # Calculate cumulative cost
                        cumulative_cost = (updated_cumulative_tokens['total_input'] * in_price_val + updated_cumulative_tokens['total_output'] * out_price_val) / 1_000_000
                        new_cumulative_cost_text = f"**Session Total Estimated Cost:** ${cumulative_cost:.6f}"

                    except ValueError:
                        error_msg = "**Estimated Cost:** Invalid prices entered"
                        new_cost_text = f"**Last Query {error_msg}"
                        new_cumulative_cost_text = f"**Session Total {error_msg}"
                        logger.warning(f"Invalid price format: Input='{current_in_price}', Output='{current_out_price}'")
                except (IndexError, KeyError, AttributeError, TypeError) as e: # Added TypeError
                    logger.warning(f"Could not parse token usage: {e} - Data: {result.get('usage')}")
                    new_token_text = "**Last Query Token Usage:** Error parsing data"
                    new_cost_text = "**Last Query Estimated Cost:** Error parsing usage data"
                    # Keep cumulative the same if current parsing fails
                    new_cumulative_token_text = cumulative_token_box.value
                    new_cumulative_cost_text = cumulative_cost_box.value
            else:
                 new_token_text = "**Last Query Token Usage:** Not available"
                 new_cost_text = "**Last Query Estimated Cost:** Usage metadata missing"
                 # Keep cumulative the same if no usage data
                 new_cumulative_token_text = cumulative_token_box.value
                 new_cumulative_cost_text = cumulative_cost_box.value
                 logger.warning("Token usage metadata not found in agent result.")

            logger.debug(f"History after invoke: {history}")

            # Clear input, update chatbot & state, and return all new texts and state
            return "", history, new_token_text, new_cost_text, new_cumulative_token_text, new_cumulative_cost_text, updated_cumulative_tokens

        # UI Layout
        with gr.Row():
             txt = gr.Textbox(
                 show_label=False,
                 placeholder="Ask a question about the catalysis knowledge graph...",
                 container=False,
                 scale=7 # Make textbox wider
             )
             submit_btn = gr.Button("Send", variant="primary", scale=1)

        # Event listeners need inputs for price textboxes and cumulative state now
        # Define inputs and outputs lists for clarity
        submit_inputs = [txt, state_hist, in_price, out_price, cumulative_tokens_state]
        submit_outputs = [txt, chatbot, token_box, cost_box, cumulative_token_box, cumulative_cost_box, cumulative_tokens_state]

        # Submit on enter key press in textbox
        txt.submit(user_submit, inputs=submit_inputs, outputs=submit_outputs)
        # Submit on button click
        submit_btn.click(user_submit, inputs=submit_inputs, outputs=submit_outputs)

        # Removed apply_prices function and button click listener
        # def apply_prices(p_in, p_out):
        #     # Just return them (no change), cost will compute next message
        #     return p_in, p_out
        # apply_price_btn.click(apply_prices, inputs=[in_price, out_price], outputs=[in_price, out_price])

    return demo

def launch_ui(
    model_name: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    debug_mode: bool,
    server_port: int = 7860,
    listen_all: bool = False, # Existing parameter
    max_history_turns: int | None = None, # New parameter for history length
    name_regex_map_path: str | None = None,
    field_regex_map_path: str | None = None
):
    """Initializes the agent and launches the Gradio UI."""
    global agent_executor_instance
    logger.info("Setting up agent executor for Gradio UI...")
    # Pass max_history_turns to setup_agent (ensure setup_agent accepts it)
    agent_executor_instance = setup_agent(
        model_name=model_name,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        verbose=debug_mode, # Use debug flag for agent verbosity
        name_regex_map_path=name_regex_map_path,
        field_regex_map_path=field_regex_map_path,
        # max_history_turns=max_history_turns # Pass the new parameter
        # ^^^ Uncomment the above line after confirming/modifying setup_agent
        # For now, just adding the param to launch_ui signature
    )

    logger.info("Building Gradio UI...")
    demo = build_ui(agent_executor_instance)

    server_name = "0.0.0.0" if listen_all else "127.0.0.1"
    logger.info(f"Launching Gradio app on {server_name}:{server_port} (Max History Turns: {max_history_turns})...")
    # Use share=True for external access if needed
    # demo.queue().launch(debug=debug_mode, server_name="0.0.0.0", server_port=server_port)
    demo.queue().launch(
        debug=debug_mode,
        server_name=server_name,
        server_port=server_port
    )

# This allows running the app directly using `python -m CATDA.ui.gradio_app`
# but it won't have the CLI arguments. Use launch_gradio.py for that.
# if __name__ == "__main__":
#     print("Warning: Running gradio_app.py directly without arguments.")
#     print("Using default settings. For configuration, use launch_gradio.py")
#     # Provide some defaults if run directly, or raise error
#     # Placeholder: replace with actual defaults or error handling
#     DEFAULT_MODEL = 'google_gemini-2.5-flash-preview-04-17'
#     NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
#     NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
#     NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
#
#     if not NEO4J_PASSWORD:
#         print("Error: NEO4J_PASSWORD environment variable not set.")
#         sys.exit(1)
#
#     # Basic logging config if run directly
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     launch_ui(
#         DEFAULT_MODEL,
#         NEO4J_URI,
#         NEO4J_USER,
#         NEO4J_PASSWORD,
#         debug_mode=True,
#         max_history_turns=10 # Example default if run directly
#     ) 