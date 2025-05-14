

# streamlit_app.py (Pro UI version)

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from context_aware import unified_analysis, save_to_excel
from datetime import datetime
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="ContextIQ", layout="wide")

# Sidebar
with st.sidebar:
    st.header("🔍 How It Works")
    st.markdown("""
    ### 🔍 Step-by-Step Breakdown

    1. **You input text** — either a single sentence or multiple ones.
    2. The system runs the text through **six NLP layers**:
        - 🧠 **Sentiment Analysis**: Detects tone — Positive, Negative, Neutral.
        - 🎭 **Emotion Detection**: Captures core emotions like Joy, Anger, Sadness.
        - 🙃 **Irony Detection**: Flags sarcasm or contradiction beneath the surface.
        - 🏷️ **NER** (Named Entity Recognition): Finds names, places, orgs, etc.
        - 🧩 **POS Tagging**: Labels each word’s grammar role (noun, verb, etc.).
        - 🚨 **Harassment Detection**: Flags content that could be sexually harassing using HateXplain fine-tuned transformers.

    3. **CHS (Context Harm Score)** is computed:
        - Based on **sentiment + emotion + irony + harassment + entity types**.
        - Uses a weighted formula to assess potential harm or toxicity.
        - Higher CHS = More likely the content is harmful, harassing, or triggering.

    4. You get:
        - **Detailed results per sentence**
        - **CHS score** & **summary table**
        - **Harassment indicator**
        - **Optional Excel/CSV export**
        - **Interactive visualizations** 🎨

    ---

    ### 🛠️ Under the Hood

    - Powered by **pre-trained transformers** and **custom scoring logic**
    - Models run **locally**, so no data is leaked 🌐❌
    - Designed for **multilingual**, **real-world** content
    """)

    st.caption("No cloud calls. Just local vibes 🌿")

# Header
st.title("💡 ContextIQ")
st.markdown("Uncover sentiment, emotion, sarcasm, and potential risk — powered by NLP & transformers")
st.caption(f"📅 Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

# Tabs for cleaner layout
tabs = st.tabs(["🔎 Analyze", "📈 Visuals", "📋 Summary Table"])

# --- TAB 1: Analyze
with tabs[0]:
    input_mode = st.radio("Choose Input Mode:", ["Single Sentence", "Multiple Sentences"], horizontal=True)

    if input_mode == "Single Sentence":
        text = st.text_area("Enter a sentence:", height=100)
        if st.button("Analyze"):
            if text.strip():
                with st.spinner("Analyzing..."):
                    result = unified_analysis(text)
                    st.success("Analysis complete")
                    st.json(result)
                    save_to_excel(result)
                    st.toast("Saved to analysis_log.xlsx")

    else:
        texts = st.text_area("Enter one sentence per line:", height=200)
        if st.button("Analyze All"):
            lines = [line.strip() for line in texts.split('\n') if line.strip()]
            if lines:
                all_results = []
                with st.spinner("Analyzing all sentences..."):
                    for idx, sentence in enumerate(lines, start=1):
                        result = unified_analysis(sentence)
                        result["text"] = sentence
                        st.markdown(f"**{idx}.** _{sentence}_")
                        st.json(result)
                        save_to_excel(result)
                        all_results.append(result)
                    st.session_state["multi_results"] = all_results
                    st.toast("All results saved to analysis_log.xlsx")

# --- TAB 2: Visuals
with tabs[1]:
    if "multi_results" in st.session_state:
        all_results = st.session_state["multi_results"]

        col1, col2 = st.columns(2)

        with col1:
            if st.checkbox("Show CHS Bar Chart"):
                texts = [res.get("text", '')[:30] + "..." for res in all_results]
                chs_scores = [res.get("chs", 0) for res in all_results]
                fig = px.bar(x=texts, y=chs_scores, labels={'x': 'Text', 'y': 'CHS Score'},
                             title="CHS Score per Sentence", color=chs_scores, color_continuous_scale='bluered')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if st.checkbox("Show Sentiment Pie"):
                sentiments = [res.get("sentiment", "Unknown") for res in all_results]
                df = pd.DataFrame(sentiments, columns=["Sentiment"])
                sentiment_counts = df["Sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Distribution")
                st.plotly_chart(fig)

        if st.checkbox("Show Emotion Bar Chart"):
            emotions = [res.get("emotion", "Unknown") for res in all_results]
            df = pd.DataFrame(emotions, columns=["Emotion"])
            emotion_counts = df["Emotion"].value_counts().reset_index()
            emotion_counts.columns = ["Emotion", "Count"]
            fig = px.bar(emotion_counts, x="Emotion", y="Count", title="Emotion Distribution",
                         color="Count", color_continuous_scale="viridis")
            st.plotly_chart(fig)

        if st.checkbox("Show Harassment Bar Chart"):
            harassment_labels = [res.get("harassment", "Unknown") for res in all_results]
            df = pd.DataFrame(harassment_labels, columns=["Harassment"])
            harassment_counts = df["Harassment"].value_counts().reset_index()
            harassment_counts.columns = ["Harassment", "Count"]
            fig = px.bar(harassment_counts, x="Harassment", y="Count",
                         title="Harassment Detection Distribution",
                         color="Count", color_continuous_scale="oranges")
            st.plotly_chart(fig)


# --- TAB 3: Summary Table
with tabs[2]:
    if "multi_results" in st.session_state:
        all_results = st.session_state["multi_results"]

        summary_data = []
        for res in all_results:
            summary_data.append({
                "Text": res.get('text', ''),
                "Translated Text": res.get('translated_text', ''),
                "Sentiment": res.get('sentiment', ''),
                "Emotion": res.get('emotion', ''),
                "Irony": res.get('irony', ''),
                "Harassment": res.get('harassment', 'Unknown'),
                "CHS": res.get('chs', 0),
                "Risk Tier": res.get('risk_tier', 'Unknown')
            })

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Summary CSV", csv, "summary.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Transformers & Streamlit ✨")
