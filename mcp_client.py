import gradio as gr
import asyncio
import os
from dotenv import load_dotenv
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
               "CRITICAL: When a user provides a file path (look for paths like /private/var/folders/... or similar), "
               "you MUST use your tools. Do NOT make assumptions or hallucinate. "
               "ALWAYS follow this exact sequence for image analysis:\n"
               "1. MUST call analyze_image_from_path with the file path to get the image description\n"
               "2. MUST call fetch_wikipedia_info with the description to get more information\n"
               "3. Provide your actual findings from the tools\n\n"
               "NEVER claim to have analyzed an image without actually calling the tools first. "
               "If a tool call fails, report the error. Do not make up results."),
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
        def get_agent_response(user_text, image_path, chat_history):
            try:
                print(f"DEBUG: Received request - Text: '{user_text}', Image: '{image_path}'")
                
                # If an image is provided, combine it with the text to form the message
                if image_path:
                    # The agent will see both the text and the path and chain the tools
                    full_message = f"{user_text} {image_path}"
                    # Add the user's turn to the chat history in Gradio 6.0 format
                    chat_history.append({"role": "user", "content": f"[Image: {image_path}]\n{user_text}"})
                    print(f"DEBUG: Processing image request with message: {full_message}")
                else:
                    # If no image, just use the text
                    full_message = user_text
                    chat_history.append({"role": "user", "content": user_text})
                    print(f"DEBUG: Processing text-only request: {full_message}")

                # Run the agent using the existing event loop
                import uuid

                # Use fresh session ID for each request
                session_id = str(uuid.uuid4())
                print(f"DEBUG: Using session ID: {session_id}")

                # Get or create event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, we need to use run_in_executor
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        response = executor.submit(
                            lambda: asyncio.run(agent.ainvoke(
                                {"messages": [HumanMessage(content=full_message)]},
                                config={"configurable": {"thread_id": session_id}}
                            ))
                        ).result()
                except RuntimeError:
                    # No running loop, safe to use asyncio.run()
                    response = asyncio.run(agent.ainvoke(
                        {"messages": [HumanMessage(content=full_message)]},
                        config={"configurable": {"thread_id": session_id}}
                    ))
                
                print(f"DEBUG: Got response: {response}")

                # The agent's final response is added to the history in Gradio 6.0 format
                if response and "messages" in response and len(response["messages"]) > 0:
                    bot_message = response["messages"][-1].content
                    print(f"DEBUG: Bot message: {bot_message}")

                    # Check if the message is empty (could indicate a malformed function call)
                    if not bot_message or bot_message.strip() == "":
                        bot_message = "I encountered an error analyzing the image. This could be due to the image size or format. Please try with a different image."
                        print("DEBUG: Empty bot message - likely a malformed function call error")
                else:
                    bot_message = "Sorry, I couldn't process your request. Please try again."
                    print("DEBUG: No valid response received")
                    
                chat_history.append({"role": "assistant", "content": bot_message})

                return "", chat_history, None # Clear textbox, return updated history, clear image box
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(f"DEBUG ERROR: {error_msg}")
                chat_history.append({"role": "assistant", "content": error_msg})
                return "", chat_history, None

        # Wire up the submit button to the handler function
        submit_btn.click(
            get_agent_response, 
            [text_box, image_box, chatbot], 
            [text_box, chatbot, image_box]
        )

    # Launch the Gradio web server.
    demo.launch(server_name="0.0.0.0")

# --- Main function to initialize agent and run web UI ---
async def main():
    """Initialize the agent and run the web UI"""
    # This setup runs only ONCE when the application starts
    print("Initializing Image Research Assistant...")
    print("Connecting to MCP servers...")

    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()
    agent = create_graph(all_tools)

    print(f"✓ Agent initialized with {len(all_tools)} tools\n")

    # Launch the web UI
    await run_ui_mode(agent)

if __name__ == "__main__":
    # Run the Image Research Assistant with web UI
    asyncio.run(main())