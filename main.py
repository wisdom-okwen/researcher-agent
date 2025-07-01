
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

gpt_api_key = os.getenv("GPT_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


gpt = ChatOpenAI(model="gpt-4o-mini", api_key="", temperature=6.0)
anthropic = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key="")