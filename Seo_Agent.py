import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

st.set_page_config(page_title="SEO Multi-Agent System", layout="wide")
st.title("SEO Multi-Agent System using AutoGen")

topic = st.text_input("Enter any topic", placeholder="Example: AI agents for business automation")
target_audience = st.text_input("Target audience", placeholder="Example: business owners, students, marketers")
content_type = st.selectbox(
    "Content type",
    ["Blog post", "LinkedIn post", "YouTube script", "Instagram carousel content"]
)

async def run_seo_agents(topic, target_audience, content_type):
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    research_agent = AssistantAgent(
        name="Research_Agent",
        model_client=model_client,
        system_message=f"""
        You are a SEO research agent.
        User topic: {topic}

        Your job:
        1. Research the topic.
        2. Find trending angles inside this topic.
        3. Identify audience pain points.
        4. Suggest primary keyword, secondary keywords, and long-tail keywords.
        5. Give 3 strong content angles.

        Output should be clear and structured.
        """
    )

    writer_agent = AssistantAgent(
        name="Content_Writer_Agent",
        model_client=model_client,
        system_message=f"""
        You are a professional content writer.

        Based on the Research Agent's output, write a complete {content_type}.
        Target audience: {target_audience}

        Make the content:
        - Simple
        - Engaging
        - Practical
        - Easy to understand
        - Conversion-focused

        Include headline, intro, body, and conclusion.
        """
    )

    seo_agent = AssistantAgent(
        name="SEO_Optimizer_Agent",
        model_client=model_client,
        system_message="""
        You are an SEO optimization expert.

        Your job:
        1. Improve title for SEO.
        2. Add meta title.
        3. Add meta description.
        4. Improve keyword placement naturally.
        5. Add proper headings.
        6. Suggest slug.
        7. Add FAQ section.
        8. Avoid keyword stuffing.

        Return the improved SEO version.
        """
    )

    review_agent = AssistantAgent(
        name="Content_Reviewer_Agent",
        model_client=model_client,
        system_message="""
        You are a senior content reviewer.

        Your job:
        1. Check clarity.
        2. Check grammar.
        3. Check flow.
        4. Check whether the content is useful.
        5. Give final polished version.
        6. Add final score out of 100.
        7. Give improvement notes if needed.

        Return only the final reviewed content.
        """
    )

    team = RoundRobinGroupChat(
        participants=[
            research_agent,
            writer_agent,
            seo_agent,
            review_agent
        ],
        max_turns=4
    )

    task = f"""
    Create SEO-optimized content.

    Topic: {topic}
    Target Audience: {target_audience}
    Content Type: {content_type}

    Workflow:
    1. Research Agent researches topic and trends.
    2. Writer Agent writes content.
    3. SEO Agent optimizes content.
    4. Reviewer Agent reviews and finalizes content.
    """

    result = await team.run(task=task)
    return result.messages

if st.button("Generate SEO Content"):
    if not topic:
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Agents are working..."):
            messages = asyncio.run(run_seo_agents(topic, target_audience, content_type))

        st.success("Content generated successfully!")

        for msg in messages:
            if hasattr(msg, "source") and hasattr(msg, "content"):
                with st.expander(f"{msg.source} Output", expanded=(msg.source == "Content_Reviewer_Agent")):
                    st.markdown(msg.content)