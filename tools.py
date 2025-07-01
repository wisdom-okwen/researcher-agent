
import os
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import time

load_dotenv()

search = DuckDuckGoSearchRun()

serpapi_api_key = os.getenv("SERPAPI_API_KEY")
serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_api_key) if serpapi_api_key else None

def robust_search(query: str, max_retries: int = 3, backoff_factor: float = 1.5):
    """
    Try DuckDuckGo search with retries and exponential backoff. Fallback to SerpAPI if all fail.
    """
    for attempt in range(max_retries):
        try:
            return search.run(query)
        except Exception as e:
            if 'Ratelimit' in str(e) or 'DuckDuckGoSearchException' in str(e):
                wait = backoff_factor ** attempt
                time.sleep(wait)
            else:
                break
    # Fallback to SerpAPI
    if serpapi is None:
        return ("DuckDuckGo search failed due to rate limiting, and SerpAPI fallback is not available. "
                "Please set the SERPAPI_API_KEY environment variable in your .env file.")
    try:
        return serpapi.run(query)
    except Exception as e:
        return f"Search failed due to rate limiting and fallback also failed: {e}"

search_tool = Tool(
    name="search",
    func=robust_search,
    description="Search the web using DuckDuckGo with fallback to SerpAPI and retry logic."
)

wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)


def save_to_txt(data: str, filename: str = "research_output.txt"):
    time_stamp = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
    formatted_text = f"----- Research Output -----\n Timestamp: {time_stamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data saved to {filename} successfully."

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save research output to a text file"
)