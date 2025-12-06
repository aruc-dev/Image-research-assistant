import gradio as gr
import asyncio
import os
import argparse
from dotenv import load_dotenv
from mcp import StdioServerParameters
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Import the MultiServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables from .env file
load_dotenv()

# --- Multi-server configuration dictionary ---
# This dictionary defines all the servers the client will connect to
server_configs = {
    "wikipedia": {
        "command": "python",
        "args": ["wikipedia_server.py"], 
        "transport": "stdio",
    },
    "vision": {
        "command": "python",
        "args": ["visual_analysis_server.py"], 
        "transport": "stdio",
    }
}

# LangGraph state definition (remains the same)
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# --- LLM Configuration ---
def get_llm():
    """Get the configured Gemini LLM from environment variables"""
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    print(f"Using Google Gemini with model: {model}")
    return ChatGoogleGenerativeAI(
        model=model, 
        temperature=0, 
        google_api_key=api_key,
        max_retries=0  # Disable retries to avoid verbose logging
    )

# --- 'create_graph' now accepts the list of tools directly ---
def create_graph(tools: list):
    # LLM configuration - using gemini-2.0-flash-exp for better accuracy and availability
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    # --- Updated system prompt to reflect new capabilities ---
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant with access to image analysis and Wikipedia research tools. "
               "When a user provides a file path (especially one ending in .jpg, .jpeg, .png, .gif, .bmp, etc.), "
               "automatically treat this as a request to analyze that image. Use your tools in this sequence:\n"
               "1. Load the image from the provided path\n"
               "2. Analyze and describe what you see in the image\n"
               "3. Research relevant topics on Wikipedia based on what you found\n"
               "4. Provide a comprehensive response with both your image analysis and research findings\n\n"
               "For other requests, use your tools appropriately to provide helpful research and information."),        
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node (remains the same)
    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    # Build LangGraph with tool routing (remains the same)
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())

# --- CLI Chat Mode ---
async def run_cli_mode(agent):
    """Interactive CLI mode for chatting with the assistant"""
    print("\n" + "="*60)
    print("Image Research Assistant - CLI Mode")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question normally for research queries")
    print("  - Provide a file path to analyze an image (e.g., /path/to/image.jpg)")
    print("  - Type 'quit' or 'exit' to end the session")
    print("="*60 + "\n")

    session_id = "cli-session"

    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # Process the message with the agent
            print("\n🤖 Assistant: ", end="", flush=True)

            try:
                response = await agent.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={"configurable": {"thread_id": session_id}}
                )
                
                # Print the assistant's response (simplified like working sample)
                if response and "messages" in response and len(response["messages"]) > 0:
                    last_message = response["messages"][-1]
                    
                    # Handle different message types
                    if hasattr(last_message, 'content'):
                        bot_message = last_message.content
                    elif isinstance(last_message, dict) and 'content' in last_message:
                        bot_message = last_message['content']
                    else:
                        bot_message = str(last_message)
                    
                    if bot_message and bot_message.strip():
                        print(f"\n{bot_message}")
                    else:
                        # Check for specific error conditions
                        if hasattr(last_message, 'response_metadata') and 'finish_reason' in last_message.response_metadata:
                            finish_reason = last_message.response_metadata.get('finish_reason')
                            if finish_reason == 'MALFORMED_FUNCTION_CALL':
                                print("\n⚠️ Function call error. Please try rephrasing your request.")
                            else:
                                print("\nNo response received.")
                        else:
                            print("\nNo response received.")
                else:
                    print("\nInvalid response format.")
                    
            except Exception as response_error:
                error_msg = str(response_error)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    print("\nAPI quota exceeded")
                else:
                    print(f"\nError: {response_error}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                print("\nAPI quota exceeded")
            else:
                print(f"\nError: {e}")

# --- UI Mode with Gradio ---
async def run_ui_mode(agent):
    """Launch the Gradio web UI"""
    print("The Image Research Assistant is ready and launching on a web UI.")

    # --- Gradio UI Implementation ---
    with gr.Blocks() as demo:
        gr.Markdown("# Image Research Assistant")
        chatbot = gr.Chatbot(label="Conversation", height=500)
        
        with gr.Row():
            # The gr.Image component will handle the upload
            # Setting type="filepath" is crucial, as it gives our tool a path to work with
            image_box = gr.Image(type="filepath", label="Upload an Image")
            
            # The textbox is for the user's text query.
            text_box = gr.Textbox(
                label="Ask a question about the image or a general research question.",
                scale=2 # Makes the textbox wider than the image box
            )

        submit_btn = gr.Button("Submit", variant="primary")
        
        # This function handles the agent's response
        # It now accepts an image_path from the gr.Image component
        async def get_agent_response(user_text, image_path, chat_history):
            # If an image is provided, combine it with the text to form the message
            if image_path:
                # The agent will see both the text and the path and chain the tools
                full_message = f"{user_text} {image_path}"
                # Add the user's turn to the chat history in Gradio 6.0 format
                chat_history.append({"role": "user", "content": f"[Image: {image_path}]\n{user_text}"})
            else:
                # If no image, just use the text
                full_message = user_text
                chat_history.append({"role": "user", "content": user_text})

            # The agent.ainvoke call remains the same, but now with the potentially combined message
            response = await agent.ainvoke(
                {"messages": [HumanMessage(content=full_message)]},
                config={"configurable": {"thread_id": "gradio-session"}}
            )

            # The agent's final response is added to the history in Gradio 6.0 format
            bot_message = response["messages"][-1].content
            chat_history.append({"role": "assistant", "content": bot_message})

            return "", chat_history, None # Clear textbox, return updated history, clear image box

        # Wire up the submit button to the handler function
        submit_btn.click(
            get_agent_response, 
            [text_box, image_box, chatbot], 
            [text_box, chatbot, image_box]
        )

    # Launch the Gradio web server.
    demo.launch(server_name="0.0.0.0")

# --- Main function to initialize agent and run selected mode ---
async def main(mode="ui"):
    """Initialize the agent and run in the selected mode"""
    # This setup runs only ONCE when the application starts
    print("Initializing Image Research Assistant...")
    print("Connecting to MCP servers...")

    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()
    agent = create_graph(all_tools)

    print(f"✓ Agent initialized with {len(all_tools)} tools\n")

    # Run in the selected mode
    if mode == "cli":
        await run_cli_mode(agent)
    else:
        await run_ui_mode(agent)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Image Research Assistant - Analyze images and research with Wikipedia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web UI (default)
  python mcp_client.py
  python mcp_client.py --mode ui

  # Launch CLI mode
  python mcp_client.py --mode cli
        """
    )

    parser.add_argument(
        "--mode",
        choices=["ui", "cli"],
        default="ui",
        help="Run mode: 'ui' for web interface (default), 'cli' for command-line chat"
    )

    args = parser.parse_args()

    # Run the application in the selected mode
    asyncio.run(main(mode=args.mode))