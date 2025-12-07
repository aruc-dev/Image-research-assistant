# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Image Research Assistant that combines visual analysis and Wikipedia search capabilities through a Gradio web interface. The system uses:
- **LangGraph** for orchestrating agentic workflows with tool routing
- **MCP (Model Context Protocol)** servers for modular tool integration
- **Google Gemini** models for both LLM reasoning (gemini-2.0-flash) and vision analysis (gemini-2.5-flash)
- **Gradio** for the web UI

## Architecture

The application consists of three main components:

### 1. MCP Client (`mcp_client.py`)
The orchestration layer that:
- Connects to multiple MCP servers using `MultiServerMCPClient`
- Builds a LangGraph state machine with conditional tool routing
- Provides a Gradio web interface for user interaction
- Manages conversation state with `MemorySaver` checkpointing

**Key flow**: User submits text + optional image → LangGraph agent decides which tools to call → Tools execute → Agent synthesizes final response

### 2. Visual Analysis Server (`visual_analysis_server.py`)
An MCP server providing two tools:
- `load_image_from_path`: Converts file paths to base64-encoded images with MIME type detection
- `get_image_description`: Analyzes base64 images using Gemini 2.5 Flash, returning detailed descriptions optimized for search queries

**Tool chaining**: The agent typically calls `load_image_from_path` first, then passes the base64 result to `get_image_description`

### 3. Wikipedia Server (`wikipedia_server.py`)
An MCP server providing:
- `fetch_wikipedia_info`: Searches Wikipedia and returns structured article data (title, summary, URL)

## Running the Application

Start the main application:
```bash
python mcp_client.py
```

This launches all components:
- Starts both MCP servers (wikipedia and vision) as subprocesses
- Connects to them via stdio transport
- Launches Gradio UI on `http://0.0.0.0:7860`

## MCP Server Configuration

The `server_configs` dictionary in `mcp_client.py:19-30` defines server connections:
```python
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
```

To add new MCP servers, extend this dictionary and ensure the server implements FastMCP.

## LangGraph State Machine

The agent uses a simple but powerful graph structure (`mcp_client.py:59-69`):
1. **START** → `chat_node`: LLM decides whether to use tools
2. **Conditional edge**: `tools_condition` routes to either:
   - `tool_node`: Execute tools, then loop back to `chat_node`
   - `END`: Return final answer

The conditional routing is handled automatically by LangGraph's `tools_condition`.

## Dependencies

Key packages (inferred from imports):
- `gradio` - Web UI framework
- `langgraph` - Agentic workflow orchestration
- `langchain-google-genai` - Google Gemini integration
- `langchain-mcp-adapters` - MCP client for LangChain
- `mcp` - Model Context Protocol SDK
- `wikipedia` - Wikipedia API wrapper
- `google-genai` - Google Generative AI SDK
