import asyncio

from dotenv import load_dotenv
load_dotenv()
from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent 
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

def search_weather_news(city: str) -> str:
     """..."""
     return str(DDGS().news(query=f"{city} weather today", max_results=5))


city = input("Which city? ")
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini") 

agent = AssistantAgent(name="weather_news", model_client=model_client,
                        tools=[search_weather_news], system_message="...",
                        reflect_on_tool_use=True)

async def main():
    await Console(agent.run_stream(task=f"... {city} ..."))
    await model_client.close()


asyncio.run(main())
