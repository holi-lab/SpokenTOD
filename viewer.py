"""Streamlit viewer for augmented dialogue data."""

import json
import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Augmented Dialogue Viewer",
    page_icon="🗣️",
    layout="wide",
)

# Styling
st.markdown("""
<style>
.user-turn {
    background-color: #e3f2fd;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.assistant-turn {
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.emotion-tag {
    background-color: #ffeb3b;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 0.8em;
}
.disfluency-tag {
    background-color: #ff9800;
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75em;
}
</style>
""", unsafe_allow_html=True)


def load_dialogues(file_path: Path) -> list[dict]:
    """Load dialogues from JSONL file."""
    dialogues = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                dialogues.append(json.loads(line))
    return dialogues


def render_turn(turn: dict, idx: int):
    """Render a single turn."""
    role = turn.get("role", "user")
    text = turn.get("text", "")
    tagged = turn.get("tagged", text)
    emotion = turn.get("emotion", {})
    disfluency = turn.get("disfluency", [])
    
    css_class = "user-turn" if role == "user" else "assistant-turn"
    role_icon = "👤" if role == "user" else "🤖"
    
    with st.container():
        col1, col2 = st.columns([1, 15])
        with col1:
            st.write(role_icon)
        with col2:
            # Header with emotion
            header_parts = [f"**Turn {idx + 1}** ({role})"]
            if emotion:
                token = emotion.get("token", "")
                label = emotion.get("label", "")
                header_parts.append(f"  `{token}` (label: {label})")
            st.markdown(" ".join(header_parts))
            
            # Tagged text
            if tagged != text and disfluency:
                st.code(tagged, language=None)
            else:
                st.write(text)
            
            # Disfluency info
            if disfluency:
                tags = " ".join([f"`[{d['type']}]`" for d in disfluency])
                st.caption(f"Disfluencies: {tags}")


def render_goal(goal: dict):
    """Render user goal."""
    st.subheader("🎯 User Goal")
    
    # Natural language goal with larger font
    text = goal.get("text", "")
    st.markdown(f"<p style='font-size: 1.2em; line-height: 1.6;'>{text}</p>", unsafe_allow_html=True)
    
    # Structured goal as JSON (inline)
    structured = goal.get("structured", {})
    if structured:
        st.write("**📋 Structured Goal:**")
        st.json(structured)


def main():
    st.title("🗣️ Augmented Dialogue Viewer")
    
    # Sidebar
    st.sidebar.header("📂 Data Selection")
    
    # Data directory
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="data/sample",
    )
    
    data_path = Path(data_dir)
    if not data_path.exists():
        st.error(f"Directory not found: {data_dir}")
        return
    
    # Split selection
    available_splits = [f.stem for f in data_path.glob("*.jsonl")]
    if not available_splits:
        st.error("No .jsonl files found in directory")
        return
    
    split = st.sidebar.selectbox("Split", available_splits)
    
    # Load data
    file_path = data_path / f"{split}.jsonl"
    dialogues = load_dialogues(file_path)
    
    st.sidebar.write(f"**Total dialogues:** {len(dialogues)}")
    
    # Source filter
    sources = sorted(set(d.get("source", "") for d in dialogues))
    selected_source = st.sidebar.selectbox("Filter by Source", ["All"] + sources)
    
    if selected_source != "All":
        dialogues = [d for d in dialogues if d.get("source") == selected_source]
        st.sidebar.write(f"**Filtered:** {len(dialogues)}")
    
    # Dialogue selection
    if not dialogues:
        st.warning("No dialogues to display")
        return
    
    dialogue_ids = [d.get("dialogue_id", f"idx_{i}") for i, d in enumerate(dialogues)]
    selected_idx = st.sidebar.selectbox(
        "Select Dialogue",
        range(len(dialogues)),
        format_func=lambda x: dialogue_ids[x],
    )
    
    # Display selected dialogue
    dialogue = dialogues[selected_idx]
    
    # Header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dialogue ID", dialogue.get("dialogue_id", "N/A"))
    with col2:
        st.metric("Source", dialogue.get("source", "N/A"))
    with col3:
        st.metric("Turns", len(dialogue.get("turns", [])))
    
    st.divider()
    
    # Goal
    goal = dialogue.get("goal", {})
    if goal:
        render_goal(goal)
        st.divider()
    
    # Turns
    st.subheader("💬 Conversation")
    turns = dialogue.get("turns", [])
    for i, turn in enumerate(turns):
        render_turn(turn, i)
    
    # Raw JSON
    with st.expander("🔍 Raw JSON"):
        st.json(dialogue)


if __name__ == "__main__":
    main()
