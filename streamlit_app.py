#!/usr/bin/env python3
"""
SAFI Root Cause Analysis - Streamlit Chat UI.

Front-end interface for the RCA engine providing:
- Sidebar controls for repository, SHA, SOW, and evidence inputs
- Chat-based interaction with structured RCA report rendering
- Follow-up Q&A grounded in the RCA result
"""

import json
import os
import sys
from typing import Optional

import streamlit as st

# Ensure we can import rca_engine from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rca_engine import run_rca_analysis, RCAConfig, DEFAULT_CONFIG
except ImportError as e:
    st.error(f"Failed to import rca_engine: {e}")
    st.stop()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Page configuration
st.set_page_config(
    page_title="SAFI Root Cause Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Welcome to the **SAFI Root Cause Analyzer**!\n\n"
                    "To get started, please provide the following in the sidebar:\n"
                    "1. **Target Repo URL or Path** - URL to a Git repository (e.g., `https://github.com/username/repo`) "
                    "or local path (e.g., `/path/to/repo`)\n"
                    "2. **Exact Git SHA** - The commit hash to analyze (e.g., `8f51ab2`)\n"
                    "3. **Failure Evidence** - Upload a `.txt` or `.log` file with error traces\n"
                    "4. **Statement of Work** *(optional)* - Upload a `.md` file with requirements\n\n"
                    "Then click **'Run Root Cause Analysis'** to begin."
                ),
            }
        ]
    if "last_rca_result" not in st.session_state:
        st.session_state.last_rca_result = None
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False


def read_uploaded_file(uploaded_file) -> str:
    """Read content from an uploaded Streamlit file."""
    if uploaded_file is None:
        return ""
    try:
        return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return ""


def format_rca_report(result: dict) -> str:
    """Format the RCA result as a structured Markdown report."""
    sha = result.get("sha", "N/A")
    confidence = result.get("confidence", "Unknown")
    affected = result.get("affected_component", "N/A")
    root_cause = result.get("root_cause", "N/A")
    evidence = result.get("evidence", [])
    uncertainty = result.get("uncertainty", "None")
    next_action = result.get("recommended_next_action", "N/A")

    # Confidence badge color
    confidence_colors = {
        "High": "🟢",
        "Medium": "🟡",
        "Low": "🔴",
    }
    badge = confidence_colors.get(confidence, "⚪")

    report = f"""## 📋 Root Cause Analysis Report

**Target SHA:** `{sha}` | **Confidence:** {badge} **{confidence}**

---

### 🎯 Affected Component
`{affected}`

---

### 🔬 Root Cause
{root_cause}

---

### 📎 Evidence
"""
    if evidence:
        for item in evidence:
            report += f"- `{item}`\n"
    else:
        report += "*No specific evidence cited.*\n"

    report += f"""

---

### ❓ Uncertainty
{uncertainty if uncertainty != "None" else "*No significant uncertainties noted.*"}

---

### 🛠️ Recommended Next Action
{next_action}
"""
    return report


def render_rca_result(result: dict) -> None:
    """Render the RCA result in the chat with download button."""
    report_md = format_rca_report(result)

    # Display the report as a chat message
    with st.chat_message("assistant"):
        st.markdown(report_md)

        # Download button for JSON report
        sha = result.get("sha", "unknown").replace("/", "_")[:40]
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download RCA JSON Report",
            data=json_str,
            file_name=f"rca_report_{sha}.json",
            mime="application/json",
            key=f"download_{sha}",
        )


def get_openai_client() -> Optional[OpenAI]:
    """Create and return an OpenAI-compatible client."""
    if OpenAI is None:
        return None
    config = DEFAULT_CONFIG
    client_kwargs = {}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    if config.api_key:
        client_kwargs["api_key"] = config.api_key
    try:
        return OpenAI(**client_kwargs)
    except Exception:
        return None


def build_followup_system_prompt(rca_result: dict) -> str:
    """Build system prompt for follow-up questions grounded in the RCA result."""
    return f"""You are a helpful assistant answering follow-up questions about a Root Cause Analysis.

The following RCA result is the GROUND TRUTH. Do not contradict it. Use it as factual context for all answers.

=== RCA RESULT (GROUND TRUTH) ===
{json.dumps(rca_result, indent=2)}
=== END RCA RESULT ===

Guidelines:
- Reference specific findings from the RCA result when relevant.
- If asked about the root cause, affected component, or evidence, quote directly from the RCA.
- Be concise and technical.
- If the question is outside the scope of the RCA, politely say so.
- Never invent new findings or contradict the RCA result.
"""


def handle_followup_question(user_query: str) -> None:
    """Handle a follow-up question using the OpenAI client with RCA context."""
    rca_result = st.session_state.last_rca_result
    if not rca_result:
        st.error("No RCA result available for context.")
        return

    client = get_openai_client()
    if not client:
        st.error("OpenAI client not available. Please install `openai` package and configure API credentials.")
        return

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Build messages for the follow-up
    messages = [
        {"role": "system", "content": build_followup_system_prompt(rca_result)},
    ]

    # Add chat history (last few messages for context)
    for msg in st.session_state.messages[-6:]:  # Last 6 messages for context
        if msg["role"] in ("user", "assistant") and msg.get("content"):
            # Skip the initial greeting and the RCA report message (which is very long)
            content = msg["content"]
            if "Welcome to the **SAFI Root Cause Analyzer**" not in content and "## 📋 Root Cause Analysis Report" not in content:
                messages.append({"role": msg["role"], "content": content})

    messages.append({"role": "user", "content": user_query})

    # Get response
    model_name = os.getenv("RCA_MODEL_NAME", "Nemotron 3 Ultra")
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    stream=True,
                )

                # Stream the response
                full_response = ""
                placeholder = st.empty()
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                # Add to chat history
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error generating response: {e}")


def run_analysis() -> None:
    """Execute the RCA analysis with current sidebar inputs."""
    repo_path = st.session_state.get("repo_path", "").strip()
    sha = st.session_state.get("sha", "").strip()
    evidence_file = st.session_state.get("evidence_file")
    sow_file = st.session_state.get("sow_file")

    # Validation
    if not repo_path:
        st.error("Please provide a Target Repo URL or Path.")
        return
    if not sha:
        st.error("Please provide an Exact Git SHA.")
        return
    if evidence_file is None:
        st.error("Please upload a Failure Evidence file.")
        return

    # Normalize Git URL: convert https:/ to https://
    if repo_path.startswith("https:/") and not repo_path.startswith("https://"):
        repo_path = "https://" + repo_path[7:]
    if repo_path.startswith("http:/") and not repo_path.startswith("http://"):
        repo_path = "http://" + repo_path[6:]

    # Read file contents
    evidence = read_uploaded_file(evidence_file)
    sow_content = read_uploaded_file(sow_file)

    if not evidence.strip():
        st.error("Failure Evidence file is empty.")
        return

    # Add user action to chat
    user_msg = (
        f"🚀 **Running RCA Analysis**\n"
        f"- Repo: `{repo_path}`\n"
        f"- SHA: `{sha}`\n"
        f"- Evidence: `{evidence_file.name}`\n"
        f"- SOW: `{sow_file.name if sow_file else 'Not provided'}`"
    )
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # Run analysis with spinner
    with st.chat_message("assistant"):
        with st.spinner("Analyzing repository AST, SOW, and failure traces..."):
            try:
                result = run_rca_analysis(
                    repo_path=repo_path,
                    sha=sha,
                    evidence=evidence,
                    sow_content=sow_content,
                )
                st.session_state.last_rca_result = result
                st.session_state.messages.append({"role": "assistant", "content": format_rca_report(result)})
                st.rerun()
            except Exception as e:
                error_msg = f"❌ **Analysis Failed**\n\n{str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()


def main() -> None:
    """Main Streamlit application."""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("🔧 Analysis Configuration")

        st.text_input(
            "Target Repo URL or Path",
            placeholder="https://github.com/username/repo or /path/to/repo",
            key="repo_path",
            help="URL to a Git repository or local path to a Git work tree",
        )

        st.text_input(
            "Exact Git SHA",
            placeholder="8f51ab2",
            key="sha",
            help="The exact commit hash to checkout and analyze",
        )

        st.file_uploader(
            "Statement of Work (.md)",
            type=["md"],
            key="sow_file",
            help="Optional: Upload a Markdown file with requirements/SOW",
        )

        st.file_uploader(
            "Failure Evidence (.txt, .log)",
            type=["txt", "log"],
            key="evidence_file",
            help="Required: Upload a text/log file with failure traces",
        )

        st.divider()

        # Run button
        if st.button(
            "🚀 Run Root Cause Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running,
        ):
            st.session_state.analysis_running = True
            run_analysis()
            st.session_state.analysis_running = False

        # Show status
        if st.session_state.last_rca_result:
            st.success("✅ Analysis complete!")
            result = st.session_state.last_rca_result
            st.caption(f"SHA: {result.get('sha', 'N/A')[:12]}")
            st.caption(f"Confidence: {result.get('confidence', 'N/A')}")

        st.divider()
        st.caption("SAFI RCA Tool • Powered by Nemotron 3 Ultra")

    # Main chat area
    st.title("🔍 SAFI Root Cause Analyzer")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input for follow-up questions
    if st.session_state.last_rca_result:
        user_query = st.chat_input("Ask a question about this diagnosis...")
        if user_query:
            handle_followup_question(user_query)
    else:
        st.chat_input("Run an analysis first to enable follow-up questions...", disabled=True)


if __name__ == "__main__":
    main()