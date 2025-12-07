# Image Research Assistant

An intelligent research assistant that combines visual analysis and Wikipedia search capabilities to answer questions about images. Upload an image of a landmark, artwork, or any object, and the assistant will identify it and provide detailed research from Wikipedia.

## Features

- **Visual Analysis**: Deep image analysis using Google Gemini 2.0 Flash to identify landmarks, artworks, and objects
- **Intelligent Research**: Automatic Wikipedia search based on image content
- **Streamlined Tool Pipeline**: Optimized single-step image analysis for fast, reliable results
- **Web Interface**: Clean Gradio web UI for easy interaction
- **Multi-Tool Architecture**: Built on the Model Context Protocol (MCP) for modular, extensible tool integration
- **Robust Error Handling**: Graceful handling of edge cases and malformed requests

## How It Works

1. **Upload an image** and ask a question (e.g., "What is this landmark and tell me about its history?")
2. The AI agent **analyzes the image** to identify what it contains
3. It automatically **searches Wikipedia** using the identified content
4. You receive a **comprehensive answer** combining visual understanding and research

## Prerequisites

- Python 3.11+ (recommended: Python 3.11.14)
- Google API key for Gemini models ([Get one here](https://ai.google.dev/))

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Image-research-assistant
```

2. **Create and activate a Python virtual environment:**
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Or use your default Python if it's 3.11+
python -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Or on Windows
# venv\Scripts\activate
```

3. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up your environment variables:**
Create a `.env` file in the project root:
```bash
# Google Gemini Configuration
GOOGLE_API_KEY=your-actual-google-api-key-here
GEMINI_MODEL=gemini-2.0-flash
```

Get your Google API key from [Google AI Studio](https://ai.google.dev/).

## Usage

Launch the web interface:

```bash
# Activate virtual environment
source venv/bin/activate

# Launch web interface
python mcp_client.py
```

The Gradio interface will launch at `http://localhost:7860`. Open this URL in your browser to start using the assistant.

### Example Queries

- Upload an image of the Eiffel Tower and ask: "What is this and when was it built?"
- Upload a painting and ask: "Who painted this and what style is it?"
- Upload a photo of a building and ask: "Where is this located?"
- Ask research questions without images: "Tell me about the Roman Colosseum"

## Architecture

The application uses a three-component architecture:

### 1. MCP Client (`mcp_client.py`)
- Orchestrates the agentic workflow using LangGraph
- Manages the Gradio web interface
- Connects to multiple MCP servers for tool access

### 2. Visual Analysis Server (`visual_analysis_server.py`)
- MCP server providing image analysis tools
- `analyze_image_from_path`: Combined tool that loads and analyzes images in a single step using Gemini Vision
- `load_image_from_path`: Loads and encodes images to Base64 (legacy support)
- `get_image_description`: Analyzes Base64-encoded images using Gemini Vision (legacy support)

### 3. Wikipedia Server (`wikipedia_server.py`)
- MCP server providing Wikipedia search
- `fetch_wikipedia_info`: Searches and retrieves article summaries

### Technology Stack

- **LangGraph**: Agentic workflow orchestration with conditional tool routing
- **MCP (Model Context Protocol)**: Standardized tool integration
- **Google Gemini**: AI models for reasoning and vision
- **Gradio**: Web interface framework
- **LangChain**: LLM framework and integrations

## Project Structure

```
Image-research-assistant/
├── mcp_client.py              # Main application and Gradio UI
├── visual_analysis_server.py  # Image analysis MCP server
├── wikipedia_server.py        # Wikipedia search MCP server
├── CLAUDE.md                  # AI assistant documentation
└── README.md                  # This file
```

## How the Agent Works

The system uses LangGraph to create an intelligent agent that:

1. Receives user input (text + optional image)
2. Decides which tools to use based on the query
3. Executes tools efficiently (streamlined single-step image analysis)
4. Chains additional tools as needed (e.g., image analysis → Wikipedia search)
5. Synthesizes information into a coherent response

The agent automatically determines the best tool execution strategy without hardcoded workflows.

### Recent Improvements

- **Optimized Tool Pipeline**: Replaced two-step image processing (load → analyze) with a single streamlined `analyze_image_from_path` tool for better reliability
- **Fixed Event Loop Handling**: Resolved asyncio conflicts in Gradio callbacks for stable concurrent request processing
- **Enhanced Error Handling**: Added graceful handling for malformed function calls and empty responses

## Extending the System

To add new capabilities:

1. Create a new MCP server using FastMCP
2. Implement tools using the `@mcp.tool()` decorator
3. Add server configuration to `server_configs` in `mcp_client.py`
4. The agent will automatically discover and use new tools

## Environment Variables

The application uses environment variables for configuration. All sensitive credentials are stored in a `.env` file (which is excluded from version control).

Required variables:
- `GOOGLE_API_KEY`: Your Google API key for accessing Gemini models

The `.env.example` file provides a template for configuration.

## Troubleshooting

**Server won't start**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify the `.env` file is configured with your API key
- Check that port 7860 is not already in use

**"API key not found" error**:
- Make sure you've created a `.env` file and added your Google API key
- Verify the key is valid at [Google AI Studio](https://ai.google.dev/)

**Image upload shows no response**:
- This issue has been fixed in the latest version with optimized tool pipeline and event loop handling
- If you still encounter issues, try refreshing the page and uploading a different image format (JPEG/PNG)

**Image analysis fails**:
- Verify your Google API key has access to Gemini models
- Check that the image file is not corrupted
- Try with a smaller image size (large images may take longer to process)

**Wikipedia searches fail**:
- Check your internet connection
- Ensure the `wikipedia` package is installed: `pip install wikipedia`

**UI appears stuck or frozen**:
- Refresh the browser page
- Restart the server: Stop the Python process and run `python mcp_client.py` again
- Check the terminal for any error messages

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
