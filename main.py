"""
Deep Research Engine — Multi-Agent Research Pipeline
=====================================================
4 specialized agents work in sequence to answer any research question:

  Searcher   → finds relevant information on the web
  Summariser → extracts key facts from search results
  Critic     → identifies gaps and triggers follow-up searches
  Writer     → produces a clean, structured final report

Run:
    python main.py
"""

import asyncio
import os
import sys
import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse
from agentscope.message import TextBlock

# ── output folder ─────────────────────────────────────────────────────────────
REPORTS_DIR = Path("./reports")
REPORTS_DIR.mkdir(exist_ok=True)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          Deep Research Engine — Powered by AgentScope        ║
║    Searcher → Summariser → Critic → Writer → Final Report    ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── web search tool (demo + real API instructions) ────────────────────────────

def web_search(query: str, num_results: int = 5) -> ToolResponse:
    """Search the web for information about a topic.

    Args:
        query (str):
            The search query to look up.
        num_results (int):
            Number of results to return, between 1 and 8. Default is 5.
    """
    # ── REAL IMPLEMENTATION ──────────────────────────────────────────────────
    # pip install google-search-results
    # export SERPAPI_KEY=your-key
    #
    # from serpapi import GoogleSearch
    # search = GoogleSearch({
    #     "q": query, "num": num_results,
    #     "api_key": os.environ["SERPAPI_KEY"]
    # })
    # results = search.get_dict().get("organic_results", [])
    # lines = [
    #     f"[{i+1}] {r['title']}\n    URL: {r['link']}\n    {r.get('snippet','')}"
    #     for i, r in enumerate(results[:num_results])
    # ]
    # return ToolResponse(content=[TextBlock(type="text", text="\n\n".join(lines))])
    # ─────────────────────────────────────────────────────────────────────────

    # Demo stub — realistic fake results tied to the query
    demo = [
        {
            "title": f"{query} — Comprehensive Overview 2025",
            "url":   f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "snippet": f"{query} refers to a broad topic with multiple dimensions. Key aspects include historical context, current applications, and future implications for society and industry.",
        },
        {
            "title": f"Latest Research on {query} | Nature",
            "url":   f"https://nature.com/articles/{query.replace(' ', '-').lower()}",
            "snippet": f"Recent studies show that {query} has significant impact across multiple sectors. Researchers have identified three primary factors driving current trends.",
        },
        {
            "title": f"{query}: Challenges and Opportunities",
            "url":   f"https://hbr.org/2025/{query.replace(' ', '-').lower()}",
            "snippet": f"Industry leaders highlight that {query} presents both major risks and substantial opportunities. Adoption rates have increased 40% year-on-year.",
        },
        {
            "title": f"How {query} is changing the world",
            "url":   f"https://wired.com/story/{query.replace(' ', '-').lower()}",
            "snippet": f"From healthcare to finance, {query} is reshaping industries. Experts predict a $2.3 trillion market impact by 2030.",
        },
        {
            "title": f"Critical perspectives on {query}",
            "url":   f"https://theatlantic.com/{query.replace(' ', '-').lower()}",
            "snippet": f"Not everyone agrees on the benefits of {query}. Critics point to ethical concerns, unequal access, and regulatory gaps that remain unaddressed.",
        },
    ]

    lines = [
        f"[{i+1}] {r['title']}\n    URL: {r['url']}\n    {r['snippet']}"
        for i, r in enumerate(demo[:num_results])
    ]
    note = "\n\n[DEMO MODE — add your SerpAPI key in main.py to use real search results]"
    text = f"Search results for: '{query}'\n\n" + "\n\n".join(lines) + note
    return ToolResponse(content=[TextBlock(type="text", text=text)])


def save_report(filename: str, content: str) -> ToolResponse:
    """Save the final research report to the reports folder.

    Args:
        filename (str):
            Name of the file to save (e.g. 'report.md'). Saved in reports/ folder.
        content (str):
            The full text content of the report to save.
    """
    try:
        path = REPORTS_DIR / filename
        path.write_text(content, encoding="utf-8")
        size = path.stat().st_size
        return ToolResponse(content=[TextBlock(type="text",
            text=f"Report saved to reports/{filename} ({size} bytes)")])
    except Exception as exc:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"Error saving report: {exc}")])


# ── model factory ─────────────────────────────────────────────────────────────

def make_model(temperature: float = 0.4) -> OpenAIChatModel:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY is not set.\n"
              "Export it first:\n  export OPENAI_API_KEY=sk-...\n")
        sys.exit(1)
    return OpenAIChatModel(
        model_name="gpt-4o",
        api_key=api_key,
        stream=True,
        generate_kwargs={"temperature": temperature},
    )


# ── agent factory ─────────────────────────────────────────────────────────────

def make_report_filename(question: str) -> str:
    """Generate a unique report filename from the question and current timestamp."""
    slug = question.lower()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = "_".join(slug.split()[:6])  # first 6 words
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"report_{slug}_{ts}.md"


def build_agents(question: str, report_filename: str):
    """Build all 4 research agents for a given question."""

    # Toolkit with search + save tools
    search_toolkit = Toolkit()
    search_toolkit.register_tool_function(web_search)

    writer_toolkit = Toolkit()
    writer_toolkit.register_tool_function(web_search)
    writer_toolkit.register_tool_function(save_report)

    searcher = ReActAgent(
        name="Searcher",
        sys_prompt=(
            f"You are a precise web researcher. Your only job is to search the web "
            f"and gather raw information about this research question:\n\n"
            f"'{question}'\n\n"
            "Run 3 different searches using different angles — broad overview, "
            "recent developments, and critical perspectives. "
            "Present the raw search results clearly. Do NOT interpret or summarise — "
            "just find and present the information."
        ),
        model=make_model(temperature=0.2),
        formatter=OpenAIChatFormatter(),
        toolkit=search_toolkit,
        memory=InMemoryMemory(),
        max_iters=6,
    )

    summariser = ReActAgent(
        name="Summariser",
        sys_prompt=(
            f"You are an expert at extracting and structuring key information. "
            f"The research question is:\n\n'{question}'\n\n"
            "You will receive raw search results from the Searcher. Your job is to:\n"
            "1. Extract the most important facts, statistics, and insights\n"
            "2. Organise them into clear categories (Background, Key Facts, "
            "Current Trends, Challenges, Opportunities)\n"
            "3. Note the source URL for each key point\n"
            "Be thorough but concise. Do not add opinions — only facts from the sources."
        ),
        model=make_model(temperature=0.3),
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
        max_iters=4,
    )

    critic = ReActAgent(
        name="Critic",
        sys_prompt=(
            f"You are a rigorous research critic. The research question is:\n\n"
            f"'{question}'\n\n"
            "You will receive a summary of research findings. Your job is to:\n"
            "1. Identify what is MISSING — what important angles haven't been covered?\n"
            "2. Flag any claims that seem weak or need more evidence\n"
            "3. Suggest 2 specific follow-up search queries that would fill the gaps\n"
            "4. Run those follow-up searches yourself using the web_search tool\n"
            "Be direct and critical. Your job is to make the research better."
        ),
        model=make_model(temperature=0.4),
        formatter=OpenAIChatFormatter(),
        toolkit=search_toolkit,
        memory=InMemoryMemory(),
        max_iters=6,
    )

    writer = ReActAgent(
        name="Writer",
        sys_prompt=(
            f"You are a professional research writer. The research question is:\n\n"
            f"'{question}'\n\n"
            "You will receive all research findings including the Summariser's key points "
            "and the Critic's additional findings. Your job is to:\n"
            "1. Write a comprehensive, well-structured research report in Markdown\n"
            "2. Use clear headings: Executive Summary, Background, Key Findings, "
            "Analysis, Challenges, Opportunities, Conclusion\n"
            "3. Include a Sources section at the end\n"
            f"4. Save the report using the save_report tool as '{report_filename}'\n"
            "Write for an intelligent, non-expert reader. Be clear, balanced, and insightful."
        ),
        model=make_model(temperature=0.5),
        formatter=OpenAIChatFormatter(),
        toolkit=writer_toolkit,
        memory=InMemoryMemory(),
        max_iters=6,
    )

    return searcher, summariser, critic, writer


# ── pipeline runner ───────────────────────────────────────────────────────────

def divider(label: str) -> None:
    pad = (58 - len(label)) // 2
    print(f"\n{'─' * pad} {label} {'─' * pad}\n")


async def run_research(question: str) -> None:
    """Run the full 4-agent research pipeline."""

    report_filename = make_report_filename(question)
    searcher, summariser, critic, writer = build_agents(question, report_filename)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n  Question : {question}")
    print(f"  Started  : {timestamp}")
    print(f"  Agents   : Searcher → Summariser → Critic → Writer\n")

    # ── PHASE 1: Searcher gathers raw information ─────────────────────────────
    divider("PHASE 1 — Searcher")
    search_result = await searcher(Msg(
        "user",
        f"Research this question thoroughly using multiple web searches: {question}",
        "user",
    ))

    # ── PHASE 2: Summariser extracts and structures key facts ─────────────────
    divider("PHASE 2 — Summariser")
    summary_result = await summariser(Msg(
        "user",
        f"Here are the raw search results from the Searcher:\n\n"
        f"{search_result.content}\n\n"
        f"Extract and structure the key information into clear categories.",
        "user",
    ))

    # ── PHASE 3: Critic finds gaps and does follow-up searches ────────────────
    divider("PHASE 3 — Critic")
    critic_result = await critic(Msg(
        "user",
        f"Here is the current research summary:\n\n"
        f"{summary_result.content}\n\n"
        f"Identify what is missing, flag weak claims, and run follow-up searches "
        f"to fill the gaps.",
        "user",
    ))

    # ── PHASE 4: Writer produces the final report ─────────────────────────────
    divider("PHASE 4 — Writer")
    await writer(Msg(
        "user",
        f"Here is all the research gathered so far:\n\n"
        f"=== SUMMARISER'S KEY FINDINGS ===\n{summary_result.content}\n\n"
        f"=== CRITIC'S ADDITIONAL FINDINGS & GAPS ===\n{critic_result.content}\n\n"
        f"Write a comprehensive research report and save it as '{report_filename}'.",
        "user",
    ))

    divider("RESEARCH COMPLETE")
    print(f"  Report saved to: reports/{report_filename}")
    print(f"  Open it in any Markdown viewer or text editor.\n")


# ── topic selection ───────────────────────────────────────────────────────────

PRESET_QUESTIONS = [
    "What is the current state of quantum computing and when will it become practical?",
    "How is AI transforming healthcare diagnosis and treatment?",
    "What are the main causes and solutions for global supply chain disruptions?",
    "What is the future of remote work after the pandemic?",
    "How are electric vehicles impacting the global oil industry?",
    "What are the biggest cybersecurity threats facing businesses in 2025?",
    "How is climate change affecting global food security?",
]


def pick_question() -> str:
    print(BANNER)
    print("Choose a research question:\n")
    for i, q in enumerate(PRESET_QUESTIONS, 1):
        print(f"  {i}. {q}")
    print("\n  Or type your own question and press Enter.\n")

    raw = input("Your choice: ").strip()

    if raw.isdigit() and 1 <= int(raw) <= len(PRESET_QUESTIONS):
        return PRESET_QUESTIONS[int(raw) - 1]
    elif raw:
        return raw
    else:
        return PRESET_QUESTIONS[0]


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    question = pick_question()
    asyncio.run(run_research(question))


if __name__ == "__main__":
    main()
