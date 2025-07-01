
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wikipedia_tool, save_tool

load_dotenv()

gpt_api_key = os.getenv("GPT_API_KEY")


class ResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools: list[str]


gpt = ChatOpenAI(model="gpt-4o-mini", api_key=gpt_api_key, temperature=0.6)

parser = PydanticOutputParser(pydantic_object=ResponseModel)

prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system", 
            "content": """
                            You are a research assistant that would help in writing a research paper.
                            Answer the user query concisely and accurately and use necessary tools.
                            Wrap the output in this format and provide no other information\n{format_instructions}.
                        """
        },
        {
            "role": "user", 
            "content": "{query}"
        },
        {
            "role": "placeholder",
            "content": "{chat_history}"
        },
        {
            "role": "placeholder",
            "content": "{agent_scratchpad}"
        }
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wikipedia_tool, save_tool]

agent = create_tool_calling_agent(
    llm=gpt,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("Hello, how can I help you with your research?  ")
raw_response = agent_executor.invoke({"query": query})

print("\n")

try:
    structured_response = parser.parse(raw_response.get("output"))
    print("\n")
    print(structured_response)

    print("\n")
except Exception as e:
    print(f"Error parsing response: {e}")