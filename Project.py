import pandas as pd
import streamlit as st
import plotly.express as px
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# 🧠 AI MODEL INITIALIZATION --------------------------------------------------

@st.cache_resource # Prevents reloading the model every time the app reruns (critical for performance)
def load_finbert():
    # Loading FinBERT: a BERT model pre-trained specifically on financial data
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    # Check for GPU (CUDA) availability to speed up inference, else fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_finbert()

# 🧪 SENTIMENT PROCESSING ENGINE ----------------------------------------------

def process_batch_sentiment(texts, batch_size=10):
    results = []
    labels = ['Bullish 📈', 'Bearish 📉', 'Neutral ⚖️'] # Labels corresponding to FinBERT output indices
    my_bar = st.progress(0, text="🤖 AI is reading articles...")
    total_texts = len(texts)

    for i in range(0, total_texts, batch_size):
        batch_chunk = texts[i : i + batch_size]
        # Tokenize text and move tensors to the same device (CPU/GPU) as the model
        inputs = tokenizer(batch_chunk, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        with torch.no_grad(): # Disable gradient calculation to save memory/time during inference
            outputs = model(**inputs)

        # Convert raw logits to probabilities (0.0 to 1.0)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

        for p in probs:
            label_idx = np.argmax(p) # Identify index with the highest probability
            # Custom compound score: Difference between positive (index 0) and negative (index 1)
            compound_score = p[0] - p[1]
            results.append((labels[label_idx], float(compound_score)))

        my_bar.progress(min((i + batch_size) / total_texts, 1.0))

    my_bar.empty() 
    return results

# 💹 MARKET DATA FETCHING -----------------------------------------------------

def get_live_market_data(ticker_name):
    ticker_upper = ticker_name.upper().strip()
    # Try the raw ticker first, then the Indian Stock Exchange (.NS) suffix if it fails
    tickers_to_try = [ticker_upper, f"{ticker_upper}.NS"]
    
    for symbol in tickers_to_try:
        try:
            data = yf.Ticker(symbol)
            hist = data.history(period="1d") # Fetch today's price data
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                prev_close = data.info.get('previousClose', price)
                change = ((price - prev_close) / prev_close) * 100
                return price, change, symbol
        except:
            continue
    return None, None, None

# ⚙️ CONFIGURATION & CLIENTS --------------------------------------------------

newsapi = NewsApiClient(api_key='a52cee776c054c4d867d0e7e54ff0a56')

st.set_page_config(page_title="Market Pulse Pro", layout="wide", page_icon="📈")

# 📂 SIDEBAR UI & NAVIGATION --------------------------------------------------

with st.sidebar:
    st.title("🚀 Market Pulse")
    st.markdown("---")

    query = st.text_input(
        "🎯 Target Asset", 
        key="target_asset"  
    )

    st.caption("Enter a stock ticker (AAPL), Crypto (BTC-USD).")
    st.markdown("---")

    # Initialize session state to manage user-selected tickers across interactions
    if "target_asset" not in st.session_state:
        st.session_state["target_asset"] = "Reliance Industries"

    def set_ticker(ticker):
        st.session_state["target_asset"] = ticker

    st.write("💡 **Popular Suggestions**")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.button("🍎 AAPL", on_click=set_ticker, args=("AAPL",), use_container_width=True)
    with s2:
        st.button("🎯 TCS", on_click=set_ticker, args=("TCS",), use_container_width=True)
    with s3:
        st.button("₿ BTC", on_click=set_ticker, args=("BTC-USD",), use_container_width=True)

    st.markdown("---")
    lookback = st.slider("📅 Lookback Period (Days)", min_value=1, max_value=28, value=7)
    
    # Reset analysis cache if the user searches for a different keyword
    if "last_query" not in st.session_state or query != st.session_state["last_query"]:
        st.session_state["full_analysis_df"] = None
        st.session_state["last_query"] = query

    st.markdown("---")
    st.write("**👨‍💻 Developed by:**")
    st.write("Mandar Ghule & Atharva Koditkar")

# 🔍 MAIN ANALYSIS LOGIC ------------------------------------------------------

st.title(f"📊 AI Sentiment Analysis: {query}")

# Execute the deep analysis only when the user clicks the button or if cache is empty
if st.session_state.get("full_analysis_df") is None:
    if st.button("✨ Initialize & Analyze All News", use_container_width=True):
        try:
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
                # Combine title and description for a more accurate sentiment reading
                texts_to_analyze = [f"{a['title']}. {a['description']}" for a in raw_articles]

                st.write(f"🧠 Running FinBERT AI on {len(texts_to_analyze)} headlines...")
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

                # Store result in session_state to prevent re-running AI on every UI change
                st.session_state['full_analysis_df'] = pd.DataFrame(processed)
                status.update(label="✅ Full Analysis Cached!", state="complete", expanded=False)
                st.rerun() 
        except Exception as e:
            st.error(f"❌ Error during initialization: {e}")

# 📡 LIVE METRICS COMPONENT ---------------------------------------------------

@st.fragment(run_every=15) # Reruns only this specific function every 15s to update prices without refreshing the whole app
def show_live_kpis(query_val, avg_score, article_count):
    price, change, found_symbol = get_live_market_data(query_val)
    currency = "₹" if (found_symbol and found_symbol.endswith(".NS")) else "$"
    
    m0, m1, m2, m3 = st.columns(4)
    with m0:
        if price:
            st.metric(f"💰 {query_val} Price", f"{currency}{price:.2f}", f"{change:.2f}%")
        else:
            st.metric("💰 Live Price", "Not Found")
    with m1:
        st.metric("🌡️ Average Score", f"{avg_score:.2f}")
    with m2:
        # Determine mood based on numeric thresholds
        mood = "Bullish 📈" if avg_score > 0.05 else "Bearish 📉" if avg_score < -0.05 else "Neutral ☁️"
        st.metric("🔮 Market Mood", mood)
    with m3:
        st.metric("📝 Articles", article_count)
    st.caption(f"Last Price Sync: {datetime.now().strftime('%M:%S')} (Auto-refreshes every 15s)")

# 🖼️ DATA VISUALIZATION DASHBOARD ---------------------------------------------

if st.session_state.get('full_analysis_df') is not None:
    full_df = st.session_state['full_analysis_df']
    # Filter the cached full dataset based on the sidebar slider (lookback)
    cutoff_date = datetime.now() - timedelta(days=lookback)
    df = full_df[full_df['Date'] >= cutoff_date].copy()

    if df.empty:
        st.warning(f"No articles found in the last {lookback} days within the analyzed batch.")
    else:
        display_df = df.copy()
        display_df['Date_Str'] = display_df['Date'].dt.strftime('%b %d')
        avg_score_val = display_df['Score'].mean()

        show_live_kpis(query, avg_score_val, len(display_df))

        tab1, tab2, tab3 = st.tabs(["📈 Momentum Trend", "📊 Distribution", "📰 News Feed"])

        with tab1:
            st.subheader("Sentiment Trend (Timeline)")
            trend_df = display_df.groupby('Date_Str')['Score'].mean().reset_index()
            fig_line = px.line(trend_df, x='Date_Str', y='Score', template="plotly_dark", markers=True)
            # Add horizontal guides for visual sentiment thresholds
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_line.add_hline(y=1, line_dash="dash", line_color="green")
            fig_line.add_hline(y=-1, line_dash="dash", line_color="red")
            fig_line.update_layout(yaxis_range=[-1.1, 1.1], xaxis_fixedrange=True, yaxis_fixedrange=True, xaxis_title="Date", yaxis_title="Avg Sentiment")
            st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

        with tab2:
            st.subheader("Sentiment Volume Split")
            fig_pie = px.pie(display_df, names='Sentiment', color='Sentiment',
                           color_discrete_map={'Bullish 📈':'#2ECC71', 'Bearish 📉':'#E74C3C', 'Neutral ⚖️':'#BDC3C7'},
                           template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            st.subheader("AI-Ranked News Feed")
            # Loop through the dataframe to create clickable news cards
            for _, row in display_df.iterrows():
                icon = "🟢" if "Bullish" in row['Sentiment'] else "🔴" if "Bearish" in row['Sentiment'] else "⚪"
                with st.expander(f"{icon} {row['Title']}"):
                    st.write(f"**🏢 Source:** {row['Source']} | **📅 Date:** {row['Date'].date()}")
                    st.write(f"🔗 [Read Full Article]({row['Link']})")
