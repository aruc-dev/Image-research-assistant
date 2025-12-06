import wikipedia
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any
import logging

mcp = FastMCP("WikipediaSearch")

@mcp.tool()
def fetch_wikipedia_info(query: str, num_articles: int = 1) -> List[Dict[str, Any]]:
    """
    Searches Wikipedia for a given query and returns the title, URL, and a brief
    summary for the top matching articles.

    Args:
        query: The search term or topic to look up (e.g., "Eiffel Tower").
        num_articles: The number of top articles to return. Defaults to 1.

    Returns:
        A list of dictionaries, where each dictionary contains the 'title',
        'summary', and 'url' of an article. Returns a list with an error
        dictionary if something goes wrong.
    """
    try:
        # Get a list of potential page titles from the search
        search_results = wikipedia.search(query, results=num_articles)
        if not search_results:
            return [{"error": f"No Wikipedia articles found for the query: '{query}'"}]

        articles_info = []
        for title in search_results:
            try:
                # Retrieve the page object for each title
                page = wikipedia.page(title, auto_suggest=False)
                articles_info.append({
                    "title": page.title,
                    "summary": page.summary.split('\n')[0],  # Get the first paragraph as a brief summary
                    "url": page.url
                })
            except wikipedia.DisambiguationError:
                # If a title is ambiguous, we'll just skip it and try the next one
                continue
            except wikipedia.PageError:
                # If a specific page fails to load, skip it
                continue
        
        if not articles_info:
             return [{"error": f"Could not load page details for query: '{query}'"}]

        return articles_info

    except Exception as e:
        return [{"error": f"An unexpected error occurred: {str(e)}"}]

# Run the MCP server
if __name__ == "__main__":
    logging.getLogger("mcp").setLevel(logging.WARNING)
    mcp.run(transport="stdio")