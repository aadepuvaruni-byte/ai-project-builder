"""
AI Project Builder
------------------
A beginner-friendly Streamlit app.
You describe a project idea -> the AI returns a plan, folder layout and starter code.

How the stack fits together:
  streamlit      -> builds the web page
  requests       -> sends your idea to the AI API
  python-dotenv  -> loads your secret API key from the .env file
  pandas         -> saves and shows your project history
  numpy          -> calculates the difficulty score
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv(override=True)  # reads the .env file

API_KEY = os.getenv("AI_API_KEY")
API_URL = os.getenv("AI_API_URL")
MODEL = os.getenv("AI_MODEL")

# ---------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------
load_dotenv()  # reads the .env file


BASE_DIR = Path(__file__).parent
HISTORY_FILE = BASE_DIR / "history.csv"
CSS_FILE = BASE_DIR / "style.css"

HISTORY_COLUMNS = ["time", "idea", "type", "language", "level", "difficulty", "plan"]

PROJECT_TYPES = ["Web app", "Data dashboard", "Chatbot", "Automation script", "Simple game"]
LANGUAGES = ["Python", "JavaScript"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]

SYSTEM_PROMPT = (
    "You are a friendly coding mentor. You explain things in plain, simple words "
    "and you never assume the student already knows technical terms. "
    "Keep the code short, commented and easy to run."
)

st.set_page_config(page_title="AI Project Builder", page_icon="🛠️", layout="wide")


# ---------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------
def load_css():
    """Read style.css and inject it into the page."""
    css = CSS_FILE.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def difficulty_score(level, feature_count):
    """Return a score from 1 to 5 using numpy.
    A higher level and more features make the project harder."""
    level_weight = {"Beginner": 1, "Intermediate": 3, "Advanced": 5}[level]
    feature_weight = min(feature_count, 10) / 2  # 0 to 5
    score = np.mean([level_weight, feature_weight])
    return float(np.clip(score, 1, 5))


def build_prompt(idea, project_type, language, level, features):
    """Turn the form answers into one clear instruction for the AI."""
    feature_text = features.strip() or "Pick a few sensible core features."
    return f"""Help me build this project.

Idea: {idea}
Project type: {project_type}
Language: {language}
My skill level: {level}
Features I want:
{feature_text}

Reply in Markdown with exactly these sections:
## 1. Project summary
Two or three simple sentences.
## 2. Tools to use
A short list, with one line explaining why for each tool.
## 3. Folder structure
A code block showing the files and folders.
## 4. Step-by-step roadmap
Numbered steps I can follow one at a time.
## 5. Starter code
The main file, with comments on every important line.
## 6. How to run it
Exact commands to type.
## 7. Next ideas
Three ways to make the project better later."""

def ask_ai(prompt):
    """Send the prompt to any OpenAI-style chat API using requests.
    Returns (text, error). One of them is always None."""
    if not API_KEY or not API_URL or not MODEL:
        return None, "Missing settings. Add AI_API_KEY, AI_API_URL and AI_MODEL to your .env file."

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "max_tokens": 3500,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(API_URL, headers=headers, json=body, timeout=120)
    except requests.exceptions.RequestException as error:
        return None, f"Could not reach the AI service: {error}"

    if response.status_code != 200:
        return None, f"The AI service returned error {response.status_code}: {response.text[:300]}"

    data = response.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, f"Unexpected reply from the AI service: {str(data)[:300]}"
    return text, None

def load_history():
    """Read saved projects from history.csv (or return an empty table)."""
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_to_history(row):
    """Add one new row to history.csv."""
    new_row = pd.DataFrame([row])
    history = load_history()
    history = new_row if history.empty else pd.concat([history, new_row], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)


# ---------------------------------------------------------------
# 3. Page layout
# ---------------------------------------------------------------
load_css()

st.markdown(
    """
    <div class="hero">
        <h1>AI Project Builder</h1>
        <p>Describe your idea. Get a plan, a folder layout and starter code you can run today.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

build_tab, history_tab = st.tabs(["Build a project", "My projects"])

# ---------- Tab 1: the builder ----------
with build_tab:
    with st.form("project_form"):
        idea = st.text_area(
            "What do you want to build?",
            placeholder="Example: A to-do list app that reminds me about homework",
            height=100,
        )

        col1, col2, col3 = st.columns(3)
        project_type = col1.selectbox("Project type", PROJECT_TYPES)
        language = col2.selectbox("Language", LANGUAGES)
        level = col3.selectbox("Your skill level", LEVELS)

        features = st.text_area(
            "Features you want (one per line, optional)",
            placeholder="Add tasks\nMark tasks as done\nSave tasks to a file",
            height=100,
        )

        submitted = st.form_submit_button("Build my project plan")

    if submitted:
        if len(idea.strip()) < 10:
            st.warning("Please describe your idea in at least a short sentence.")
        else:
            feature_list = [line for line in features.splitlines() if line.strip()]
            score = difficulty_score(level, len(feature_list))

            with st.spinner("Building your plan. This takes about 20 seconds..."):
                plan, error = ask_ai(build_prompt(idea, project_type, language, level, features))

            if error:
                st.error(error)
            else:
                st.session_state["plan"] = plan
                st.session_state["score"] = score
                st.session_state["idea"] = idea
                save_to_history(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "idea": idea.strip(),
                        "type": project_type,
                        "language": language,
                        "level": level,
                        "difficulty": round(score, 1),
                        "plan": plan,
                    }
                )

    # Show the latest result (it stays on screen after the download button is used)
    if "plan" in st.session_state:
        score = st.session_state["score"]
        st.markdown(f'<p class="meter-label">Difficulty: {score:.1f} out of 5</p>', unsafe_allow_html=True)
        st.progress(score / 5)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(st.session_state["plan"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "Download plan (.md)",
            data=st.session_state["plan"],
            file_name="project-plan.md",
            mime="text/markdown",
        )

# ---------- Tab 2: saved projects ----------
with history_tab:
    history = load_history()

    if history.empty:
        st.info("No projects yet. Build your first one in the other tab.")
    else:
        st.dataframe(
            history[["time", "idea", "type", "language", "level", "difficulty"]],
            use_container_width=True,
            hide_index=True,
        )

        choice = st.selectbox(
            "Open a saved plan",
            options=history.index[::-1],  # newest first
            format_func=lambda i: f"{history.loc[i, 'time']} - {str(history.loc[i, 'idea'])[:50]}",
        )
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(str(history.loc[choice, "plan"]))
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Clear all history"):
            HISTORY_FILE.unlink(missing_ok=True)
            st.rerun()