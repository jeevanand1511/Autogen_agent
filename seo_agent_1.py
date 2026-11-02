import os
import asyncio
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Please configure your .env file.")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=API_KEY,
)


# ==========================================
# 2. WEB SEARCH TOOL
# ==========================================

def search_web(query: str) -> str:
    """
    Search recent web results for SEO research.
    """

    try:
        results = DDGS().text(
            query,
            region="in-en",
            timelimit="m",
            max_results=8,
        )

        if not results:
            return "No search results found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            formatted_results.append(
                f"""
                Result {index}
                Title: {result.get('title', 'N/A')}
                Description: {result.get('body', 'N/A')}
                URL: {result.get('href', 'N/A')}
                """
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Web search failed: {str(e)}"


# ==========================================
# 3. CREATE AGENT 1 - RESEARCH AGENT
# ==========================================

research_agent = AssistantAgent(
    name="SEO_Research_Agent",
    model_client=model_client,
    tools=[search_web],
    reflect_on_tool_use=True,
    system_message="""
    You are an expert SEO Research Analyst.

    Your responsibilities:

    1. Research the topic provided by the user.
    2. Use the search_web tool to find recent information.
    3. Identify trending angles and emerging discussions.
    4. Find primary and secondary keywords.
    5. Identify user search intent.
    6. Identify content gaps and opportunities.
    7. Provide relevant source URLs.

    Focus on recent search results, preferably from the
    last month.

    Do not invent search volume or keyword difficulty.

    Clearly distinguish verified findings from your
    own recommendations.

    Return a structured research report.
    """,
)


# ==========================================
# 4. CREATE AGENT 2 - CONTENT WRITER
# ==========================================

content_writer_agent = AssistantAgent(
    name="SEO_Content_Writer_Agent",
    model_client=model_client,
    system_message="""
    You are an expert SEO Content Writer.

    Your responsibilities:

    1. Read the research report from the previous agent.
    2. Select the most useful and relevant topic angle.
    3. Write a detailed, engaging SEO-friendly article.
    4. Include an attractive H1 title.
    5. Organize content using H2 and H3 headings.
    6. Naturally include primary and secondary keywords.
    7. Include an introduction and conclusion.
    8. Include 5 frequently asked questions.
    9. Write original, human-readable content.

    Avoid keyword stuffing and unsupported claims.

    Return the complete article in Markdown format.
    """,
)


# ==========================================
# 5. CREATE AGENT 3 - SEO OPTIMIZER
# ==========================================

seo_optimizer_agent = AssistantAgent(
    name="SEO_Optimizer_Agent",
    model_client=model_client,
    system_message="""
    You are a professional Technical SEO Specialist.

    Your responsibilities:

    1. Review the article written by the content writer.
    2. Check primary keyword placement.
    3. Check secondary keyword usage.
    4. Optimize the SEO title.
    5. Write a compelling meta description.
    6. Check heading hierarchy.
    7. Improve readability.
    8. Check URL slug recommendation.
    9. Recommend internal and external links.
    10. Avoid keyword stuffing.

    Return:

    - Optimized SEO title
    - Meta description
    - Primary keyword
    - Secondary keywords
    - Recommended URL slug
    - Optimized article
    - SEO improvement notes

    Preserve factual accuracy and source references.
    """,
)


# ==========================================
# 6. CREATE AGENT 4 - CONTENT REVIEWER
# ==========================================

content_reviewer_agent = AssistantAgent(
    name="Content_Review_Agent",
    model_client=model_client,
    system_message="""
    You are a Senior Content Quality Reviewer.

    Your responsibilities:

    1. Review the SEO-optimized article.
    2. Check grammar and spelling.
    3. Check sentence structure.
    4. Identify repetitive content.
    5. Check clarity and readability.
    6. Identify unsupported factual claims.
    7. Check whether the article matches search intent.
    8. Ensure the content is useful to readers.
    9. Correct identified issues.

    Return the final corrected article.

    Also provide:

    - Content quality score out of 100.
    - SEO quality observations.
    - Grammar and readability observations.
    - Final approval status.

    Never claim that an article is guaranteed to rank
    on Google.
    """,
)


# ==========================================
# 7. MULTI-AGENT WORKFLOW
# ==========================================

async def generate_seo_content(topic, progress_callback):

    outputs = {}

    try:

        # Agent 1: Research
        progress_callback("Agent 1: Researching topic...")

        research_result = await research_agent.run(
            task=f"""
            Research this topic: {topic}

            Search for recent developments and trending
            discussions related to this topic.

            Provide a structured SEO research report.
            """
        )

        outputs["research"] = research_result.messages[-1].content

        # Agent 2: Content writing
        progress_callback("Agent 2: Writing SEO content...")

        writing_result = await content_writer_agent.run(
            task=f"""
            USER TOPIC:
            {topic}

            RESEARCH REPORT:
            {outputs['research']}

            Write a complete SEO-friendly article
            based on this research.
            """
        )

        outputs["content"] = writing_result.messages[-1].content

        # Agent 3: SEO optimization
        progress_callback("Agent 3: Optimizing SEO...")

        seo_result = await seo_optimizer_agent.run(
            task=f"""
            USER TOPIC:
            {topic}

            RESEARCH:
            {outputs['research']}

            ARTICLE:
            {outputs['content']}

            Optimize this article for SEO.
            Return the complete optimized article
            and SEO metadata.
            """
        )

        outputs["seo"] = seo_result.messages[-1].content

        # Agent 4: Content review
        progress_callback("Agent 4: Reviewing final content...")

        review_result = await content_reviewer_agent.run(
            task=f"""
            USER TOPIC:
            {topic}

            RESEARCH:
            {outputs['research']}

            SEO OPTIMIZED CONTENT:
            {outputs['seo']}

            Perform the final content quality review.

            Correct any identified problems and return
            the final approved article, SEO observations,
            quality score and approval status.
            """
        )

        outputs["final"] = review_result.messages[-1].content

        return outputs

    finally:
        await model_client.close()


# ==========================================
# 8. STREAMLIT FRONTEND
# ==========================================

st.set_page_config(
    page_title="Multi-Agent SEO Generator",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Multi-Agent SEO Content Generator")

st.write(
    "Generate researched, SEO-optimized content using "
    "four specialized AutoGen AI agents."
)

st.divider()

topic = st.text_input(
    "Enter your topic",
    placeholder="Example: Artificial Intelligence in Healthcare",
)

generate_button = st.button(
    "Generate SEO Content",
    type="primary",
    use_container_width=True,
)

if generate_button:

    if not topic.strip():

        st.warning("Please enter a topic.")

    else:

        progress_messages = []

        status = st.status(
            "Starting SEO multi-agent workflow...",
            expanded=True,
        )

        def update_progress(message):
            progress_messages.append(message)
            status.write("⏳ " + message)

        try:

            outputs = asyncio.run(
                generate_seo_content(
                    topic,
                    update_progress,
                )
            )

            status.update(
                label="SEO content generation completed!",
                state="complete",
                expanded=False,
            )

            st.success("All four agents completed successfully!")

            # Results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                [
                    "Research",
                    "Content",
                    "SEO Optimization",
                    "Final Review",
                    "Download",
                ]
            )

            with tab1:
                st.subheader("Agent 1 - SEO Research Report")
                st.markdown(outputs["research"])

            with tab2:
                st.subheader("Agent 2 - Generated Article")
                st.markdown(outputs["content"])

            with tab3:
                st.subheader("Agent 3 - SEO Optimization")
                st.markdown(outputs["seo"])

            with tab4:
                st.subheader("Agent 4 - Final Review")
                st.markdown(outputs["final"])

            with tab5:

                st.subheader("Download SEO Report")

                complete_report = f"""
# SEO MULTI-AGENT REPORT

Topic: {topic}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

# 1. RESEARCH REPORT

{outputs['research']}

# 2. CONTENT

{outputs['content']}

# 3. SEO OPTIMIZATION

{outputs['seo']}

# 4. FINAL REVIEW

{outputs['final']}
"""

                st.download_button(
                    label="Download Complete SEO Report",
                    data=complete_report,
                    file_name="seo_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        except Exception as e:

            status.update(
                label="Workflow failed",
                state="error",
            )

            st.error(f"Error: {str(e)}")

            st.info(
                "Check your API key, internet connection "
                "and installed package versions."
            )