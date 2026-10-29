import asyncio
from dotenv import load_dotenv
load_dotenv()
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
agent = AssistantAgent(name="hr_assistant", model_client=model_client,
 system_message="You are an HR assistant.")
async def main():
 await Console(agent.run_stream(task="..."))
 await model_client.close()
asyncio.run(main())
