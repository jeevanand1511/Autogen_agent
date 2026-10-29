
import asyncio
import os
import json

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Load environment variables from .env
load_dotenv()

st.set_page_config(
    page_title="AutoGen Weather & News Agent",
    page_icon="🌦️",
    layout="centered"
)


# DuckDuckGo news search tool
def search_weather_news(city: str) -> str:
    """Search for recent weather-related news for a city."""
    try:
        results = DDGS().news(
            query=f"{city} weather today",
            max_results=5
        )

        if not results:
            return f"No weather news found for {city}."

        return json.dumps(results, ensure_ascii=False, default=str)

    except Exception as e:
        return f"News search failed: {str(e)}"


# Run the AutoGen agent asynchronously
async def run_weather_agent(city: str) -> str:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    try:
        agent = AssistantAgent(
            name="weather_news",
            model_client=model_client,
            tools=[search_weather_news],
            system_message=(
                "You are a helpful weather news assistant. "
                "Use the search_weather_news tool to find relevant "
                "weather news for the requested city. "
                "Summarize the results clearly. "
                "Distinguish reported news from verified weather forecasts. "
                "If search results are unavailable, explain that clearly."
            ),
            reflect_on_tool_use=True
        )

        result = await agent.run(
            task=(
                f"Find the latest available weather-related news "
                f"for {city}. Summarize the important findings "
                f"and include source links when available."
            )
        )

        # Get the final assistant response
        for message in reversed(result.messages):
            if (
                getattr(message, "source", None) == agent.name
                and isinstance(getattr(message, "content", None), str)
            ):
                return message.content

        return "The agent completed without returning a text response."

    finally:
        await model_client.close()


# Streamlit user interface
st.title("🌦️ AutoGen Weather & News Agent")

st.write(
    "Enter a city to search for recent weather-related news "
    "using DuckDuckGo and summarize it with AutoGen."
)

city = st.text_input(
    "Enter city name",
    placeholder="Example: Chennai"
)

if st.button("Search Weather News", type="primary"):
    if not city.strip():
        st.warning("Please enter a city name.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is missing. Add it to your .env file "
            "and restart the app."
        )
    else:
        try:
            with st.spinner(
                f"Searching weather news for {city.strip()}..."
            ):
                answer = asyncio.run(
                    run_weather_agent(city.strip())
                )

            st.subheader(f"Weather News: {city.strip()}")
            st.markdown(answer)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info(
                "Check your OpenAI API key, internet connection, "
                "and installed package versions."
            )

st.caption(
    "Powered by Streamlit, AutoGen, OpenAI, and DuckDuckGo."
)
