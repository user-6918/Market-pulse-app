import pandas as pd
import streamlit as st
import plotly.express as px
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime, timedelta

# --- 🧠 SECTION 1: AI MODEL CORE ---
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_finbert()

def process_batch_sentiment(texts, batch_size=10):
    results = []
    labels = ['Bullish 📈', 'Bearish 📉', 'Neutral ⚖️']
    my_bar = st.progress(0, text="🤖 AI is reading articles...")
    total_texts = len(texts)
    
    for i in range(0, total_texts, batch_size):
        batch_chunk = texts[i : i + batch_size]
        inputs = tokenizer(batch_chunk, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        for p in probs:
            label_idx = np.argmax(p)
            compound_score = p[0] - p[1]
            results.append((labels[label_idx], float(compound_score)))
            
        my_bar.progress(min((i + batch_size) / total_texts, 1.0))
        
    my_bar.empty() 
    return results

# --- 🌐 SECTION 2: DATA ACQUISITION ---
# Note: Ensure your API key is valid
newsapi = NewsApiClient(api_key='a52cee776c054c4d867d0e7e54ff0a56')

# --- 🎨 SECTION 3: UI CONFIGURATION ---
st.set_page_config(page_title="Market Pulse Pro", layout="wide", page_icon="📈")

# --- 🛠️ SECTION 4: CONTROL CENTER (SIDEBAR) ---
with st.sidebar:
    st.title("🚀 Market Pulse")
    st.markdown("---")
    
    query = st.text_input("🎯 Target Asset", value="Reliance Industries")
    st.caption("Enter a stock, commodity, or index to analyze related news sentiment.")
    st.markdown("---")
    
    lookback = st.slider("📅 Lookback Period (Days)", min_value=1, max_value=28, value=7)
    st.caption("Adjust the slider to filter news articles by days.")
    
    # Reset full cache only if the KEYWORD changes
    if "last_query" not in st.session_state or query != st.session_state["last_query"]:
        st.session_state["full_analysis_df"] = None
        st.session_state["last_query"] = query

    st.markdown("---")
    st.write("**👨‍💻 Developed by:**")
    st.write("Mandar Ghule & Atharva Koditkar")

# --- ⚡ SECTION 5: THE ENGINE (FETCH & ANALYZE ONCE) ---
st.title(f"📊 AI Sentiment Analysis: {query}")

# We fetch the maximum range initially to "prime" the app
if st.session_state.get("full_analysis_df") is None:
    if st.button("✨ Initialize & Analyze All News", use_container_width=True):
        try:
            # Set start date to 28 days ago to capture maximum history
            max_start = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
            
            with st.status(f"🛰️ Scanning the web for {query}...", expanded=True) as status:
                st.write("📥 Fetching latest news...")
                articles_data = newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    from_param=max_start,
                    page_size=100
                )

                if not articles_data['articles']:
                    st.warning("⚠️ No news found. Try a broader search term.")
                    st.stop()

                raw_articles = articles_data['articles']
                texts_to_analyze = [f"{a['title']}. {a['description']}" for a in raw_articles]
                
                st.write(f"🧠 Running FinBERT AI on {len(texts_to_analyze)} headlines...")
                # Repaired: Added batch_size and fixed variable name 'raw_articles'
                sentiment_results = process_batch_sentiment(texts_to_analyze, batch_size=5)

                processed = []
                for i, article in enumerate(raw_articles):
                    sentiment, score = sentiment_results[i]
                    processed.append({
                        'Date': pd.to_datetime(article['publishedAt']).tz_localize(None),
                        'Title': article['title'],
                        'Sentiment': sentiment,
                        'Score': score,
                        'Link': article['url'],
                        'Source': article['source']['name']
                    })
                
                st.session_state['full_analysis_df'] = pd.DataFrame(processed)
                status.update(label="✅ Full Analysis Cached!", state="complete", expanded=False)
                st.rerun() 
        except Exception as e:
            st.error(f"❌ Error during initialization: {e}")

# --- 📊 SECTION 6: INSTANT FILTERING & DASHBOARD ---
if st.session_state.get('full_analysis_df') is not None:
    full_df = st.session_state['full_analysis_df']
    # Instant filtering logic based on the slider value
    cutoff_date = datetime.now() - timedelta(days=lookback)
    df = full_df[full_df['Date'] >= cutoff_date].copy()

    if df.empty:
        st.warning(f"No articles found in the last {lookback} days within the analyzed batch.")
    else:
        display_df = df.copy()
        display_df['Date_Str'] = display_df['Date'].dt.strftime('%b %d')
        avg_score = display_df['Score'].mean()

        # KPIs
        m1, m2, m3 = st.columns(3)
        m1.metric("🌡️ Average Score", f"{avg_score:.2f}")
        mood = "Bullish 📈" if avg_score > 0.05 else "Bearish 📉" if avg_score < -0.05 else "Neutral ☁️"
        m2.metric("🔮 Market Mood", mood)
        m3.metric("📝 Articles in Window", len(display_df))

        tab1, tab2, tab3 = st.tabs(["📈 Momentum Trend", "📊 Distribution", "📰 News Feed"])
        
        with tab1:
            st.subheader("Sentiment Trend (Timeline)")
            trend_df = display_df.groupby('Date_str')['Score'].mean().reset_index()
            fig_line = px.line(trend_df, x='Date_str', y='Score', template="plotly_dark", markers=True)
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_line.add_hline(y=1, line_dash="dash", line_color="green")
            fig_line.add_hline(y=-1, line_dash="dash", line_color="red")
            fig_line.update_layout(yaxis_range=[-1.1, 1.1],xaxis_fixedrange=True,yaxis_fixedrange=True)
            st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

        with tab2:
            st.subheader("Sentiment Volume Split")
            fig_pie = px.pie(display_df, names='Sentiment', color='Sentiment',
                           color_discrete_map={'Bullish 📈':'#2ECC71', 'Bearish 📉':'#E74C3C', 'Neutral ⚖️':'#BDC3C7'},
                           template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            st.subheader("AI-Ranked News Feed")
            for _, row in display_df.iterrows():
                icon = "🟢" if "Bullish" in row['Sentiment'] else "🔴" if "Bearish" in row['Sentiment'] else "⚪"
                with st.expander(f"{icon} {row['Title']}"):
                    st.write(f"**🏢 Source:** {row['Source']} | **📅 Date:** {row['Date'].date()}")
                    st.write(f"🔗 [Read Full Article]({row['Link']})")
