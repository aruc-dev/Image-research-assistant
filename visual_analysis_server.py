import os
import base64
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from typing import List
from google import genai
from google.genai import types
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize the FastMCP server
mcp = FastMCP("VisualAnalysisServer")

@mcp.tool()
def get_image_description(base64_image_string: str, mime_type: str) -> str:
    """
    Performs a deep analysis of a Base64 encoded image and returns a detailed,
    descriptive paragraph about its content. If the image is of a known landmark,
    it will be specifically identified. This description is intended to be used
    as a high-quality search query for a research tool.

    Args:
        base64_image_string: The image file encoded as a Base64 string.
        mime_type: The MIME type of the image (e.g., "image/jpeg", "image/png").

    Returns:
        A single string containing a detailed description of the image.
        Returns an error message if analysis fails.
    """
    try:
        image_bytes = base64.b64decode(base64_image_string)

        # This explicitly creates the image part of the prompt using the official SDK type
        image_part = types.Part.from_bytes(
            mime_type=mime_type,
            data=image_bytes
        )
        
        # --- prompt text ---
        prompt_text = (
            "Analyze this image in detail. Provide a concise, one-paragraph description. "
            "If it is a famous landmark, work of art, or specific location, identify it by name. "
            "Focus on the most important and defining elements in the image that would be useful for a web search. "
            "For example, instead of 'a building', say 'the Eiffel Tower in Paris'. "
            "Do not add any conversational filler; return only the description."
        )

        model = genai.Client('gemini-2.5-flash', google_api_key=os.getenv("GOOGLE_API_KEY"))
        # The generate_content call now uses the correctly typed image_part object
        response = model.generate_content([image_part, prompt_text])
        
        # --- Return the single description string directly ---
        # The .strip() removes any leading/trailing newlines or spaces from the model's output
        description = response.text.strip()
        
        return description

    except Exception as e:
        return f"Error analyzing image: {e}"
        
@mcp.tool()
def load_image_from_path(file_path: str) -> dict:
    """
    Loads an image from a server-accessible file path, encodes it to Base64,
    and determines its MIME type.

    Args:
        file_path: The absolute path to the image file, which must be
                   accessible by the server running this tool.

    Returns:
        A dictionary containing the 'base64_image_string' and 'mime_type',
        or an 'error' key if loading fails.
    """
    try:
        image_path = Path(file_path)
        if not image_path.is_file():
            return {"error": f"File not found at path: {file_path}"}

        # Open the file in binary read mode
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Encode the binary data to a Base64 string
        base64_string = base64.b64encode(image_data).decode("utf-8")
        
        # Guess the MIME type from the file extension
        mime_type, _ = mimetypes.guess_type(image_path)
        
        if not mime_type:
            mime_type = "application/octet-stream"  # A generic default
            
        return {
            "base64_image_string": base64_string,
            "mime_type": mime_type
        }
    except FileNotFoundError:
        return {"error": f"File not found at path: {file_path}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred while loading the image: {str(e)}"}

if __name__ == "__main__":
    logging.getLogger("mcp").setLevel(logging.WARNING)
    mcp.run(transport="stdio")