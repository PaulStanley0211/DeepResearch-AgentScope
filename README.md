# Deep Research Agent

A multi-agent research pipeline powered by [AgentScope](https://github.com/modelscope/agentscope) and GPT-4o. Given any research question, it autonomously searches the web, synthesizes findings, critiques gaps, and produces a structured Markdown report — saved automatically to disk.

---

## How It Works

Four specialized AI agents work in sequence:

```
Searcher → Summariser → Critic → Writer → report saved to reports/
```

| Agent | Role |
|---|---|
| **Searcher** | Runs multiple web searches from different angles (broad overview, recent developments, critical perspectives) and returns raw results |
| **Summariser** | Extracts key facts, statistics, and insights from raw results and organizes them into structured categories |
| **Critic** | Identifies gaps and weak claims in the summary, then runs follow-up searches to fill them |
| **Writer** | Synthesizes all findings into a polished, well-structured Markdown research report and saves it |

Each run produces a **unique report file** named after the question and timestamp — previous reports are never overwritten.

---

## What It Researches

The agent can answer any open-ended research question. Built-in preset topics include:

- Current state and future of **quantum computing**
- How **AI is transforming healthcare** diagnosis and treatment
- Causes and solutions for **global supply chain disruptions**
- The future of **remote work** post-pandemic
- Impact of **electric vehicles** on the global oil industry
- Top **cybersecurity threats** facing businesses
- How **climate change** is affecting global food security

You can also type any custom question at runtime.

---

## Report Structure

Every generated report follows this structure:

1. **Executive Summary** — high-level answer to the question
2. **Background** — context and history
3. **Key Findings** — facts, statistics, and data points
4. **Analysis** — interpretation of the findings
5. **Challenges** — risks, barriers, and open problems
6. **Opportunities** — potential upside and future directions
7. **Conclusion** — synthesized takeaway
8. **Sources** — URLs cited throughout

Reports are saved to `Deep_research/reports/` as:
```
report_<question-slug>_<YYYYMMDD_HHMMSS>.md
```

---

## Setup

### 1. Prerequisites

- Python 3.14+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root (never commit this file):

```env
OPENAI_API_KEY=sk-...
SERP_API_KEY=...        # optional — only needed for real web search
```

### 4. Run

```bash
cd Deep_research
python main.py
```

---

## Web Search Modes

| Mode | Description |
|---|---|
| **Demo mode** (default) | Generates realistic simulated search results — no API key required. Good for testing the pipeline. |
| **Real mode** | Uses [SerpAPI](https://serpapi.com) to fetch live Google results. Requires uncommenting the real implementation in `main.py` and a valid `SERP_API_KEY`. |

To enable real search, install the SerpAPI client and uncomment the block in [Deep_research/main.py](Deep_research/main.py):

```bash
pip install google-search-results
```

---

## Project Structure

```
DeepResearchAgent/
├── Deep_research/
│   ├── main.py          # All agent logic and pipeline
│   └── reports/         # Generated report files (auto-created)
├── .env                 # API keys — never committed
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `agentscope` | Multi-agent framework (ReActAgent, Toolkit, Memory) |
| `openai` | GPT-4o model client |
| `python-dotenv` | Loads API keys from `.env` |

---

## Security

- The `.env` file is excluded from version control via `.gitignore`
- API keys are loaded at runtime from environment variables — never hardcoded
- Report filenames are sanitized to prevent path traversal

---

## License

MIT
