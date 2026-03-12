import nltk
import pandas as pd
import streamlit as st
import plotly.express as px
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER and NewsAPI
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
newsapi = NewsApiClient(api_key='a52cee776c054c4d867d0e7e54ff0a56')

# ---1.Page Configuration---
st.set_page_config(page_title="Market Pulse", layout="wide", page_icon="📈")

# ---2.Sidebar---
with st.sidebar:
    st.title("🚀 Market Pulse")
    st.markdown("---")

    # 1. Main Input
    query = st.text_input("Target Asset", value="Reliance Industries", help="Enter a stock ticker (AAPL) or commodity (Gold).")

    st.markdown("---")

    # 2. Date Range Selector
    st.markdown("### 📅 Analysis Window")
    # This creates a slider to choose how many days of news to pull
    lookback = st.slider("Lookback Period (Days)", min_value=1, max_value=28, value=7)
    st.caption(f"Analyzing headlines from the last {lookback} days.")

    st.markdown("---")

    st.write("**Developed by:**")
    st.write("Mandar Dada Ghule")
    st.write("Atharva Koditkar")

    st.markdown("---")

# ---3.Main Header---
st.title(f"📊 Sentiment Analysis: {query}")
analyze_btn = st.button("Generate Insights", use_container_width=True)

if analyze_btn:
    try:

        from datetime import datetime, timedelta

        start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')

        with st.spinner(f'Fetching news for {query} since {start_date}...'):
            # Update the fetch call with 'from_param'
            articles = newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                from_param=start_date  # Connects the slider to the API
            )

        news_list = []
        for article in articles['articles'][:100]:
            text = f"{article['title']} {article['description']}"
            score = sia.polarity_scores(text)['compound']

            if score >= 0.05:
                sentiment = 'Bullish'
            elif score <= -0.05:
                sentiment = 'Bearish'
            else:
                sentiment = 'Neutral'

            news_list.append({
                'Date': article['publishedAt'][:10],
                'Title': article['title'],
                'Sentiment': sentiment,
                'Score': score,
                'Link': article['url'],
                'Source': article['source']['name']
            })

        df = pd.DataFrame(news_list)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%b %d')

        # --- 4.TOP METRICS (Now theme-adaptive) ---
        avg_score = df['Score'].mean()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean Sentiment", f"{avg_score:.2f}")
        with col2:
            if avg_score > 0.5:
                sentiment_label = "Bullish"
            elif avg_score < -0.5:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            st.metric("Market Mood", sentiment_label)
        with col3:
            st.metric("Articles Analyzed", len(df))

        st.markdown("---")

        # --- 5.TABS ---
        tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "📊 Sentiment Split", "📰 News Feed"])

        with tab1:
            st.subheader("Sentiment Momentum Over Time")
            sentiment_over_time = df.groupby('Date')['Score'].mean().reset_index().sort_values('Date')
            fig_line = px.line(sentiment_over_time, x='Date', y='Score',template="plotly_dark", markers=True)
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_line.add_hline(y=1,line_dash="dash", line_color="#2ECC71", annotation_text="Max Bullish")
            fig_line.add_hline(y=-1,line_dash="dash", line_color="#E74C3C", annotation_text="Max Bearish")
            fig_line.update_layout(yaxis_range=[-1.1, 1.1])
            st.plotly_chart(fig_line, use_container_width=True,config={"displayModeBar": False, "scrollZoom": False})

        with tab2:
            st.subheader("Sentiment Distribution")
            fig_pie = px.pie(df, names='Sentiment', color='Sentiment',
                           color_discrete_map={'Bullish':'#2ECC71', 'Bearish':'#E74C3C', 'Neutral':'#BDC3C7'},
                           template="plotly_dark")
            st.plotly_chart(fig_pie)

        with tab3:
            st.subheader("Latest Headlines")
            for i, row in df.head(15).iterrows():
                color = "🟢" if row['Sentiment'] == 'Bullish' else "🔴" if row['Sentiment'] == 'Bearish' else "⚪"
                with st.expander(f"{color} {row['Title']} ({row['Source']})"):
                    st.write(f"**Score:** {row['Score']} | **Date:** {row['Date']}")
                    st.write(f"[Read full article]({row['Link']})")

    except Exception as e:
        st.error(f"Error: {e}")




