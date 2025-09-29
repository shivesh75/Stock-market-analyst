import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
# ============= CONFIG ====================
# Set up Gemini client (make sure GEMINI_API_KEY is exported in your env)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ============= FEATURE ENGINEERING ====================
def load_data(ticker):
    try:
        df = yf.download(ticker, period="5y")  # download stock data
        if df.empty:  # if no data returned
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
            return None

        # Flatten columns in case yfinance returns MultiIndex
        df.columns = df.columns.to_flat_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Feature engineering
        df["Return"] = df["Close"].pct_change()
        df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        df["Volatility"] = df["Return"].rolling(10).std()
        df.dropna(inplace=True)

        return df

    except Exception as e:  # catch any other errors
        st.error(f"Error downloading data for '{ticker}': {e}")
        return None


# ============= ML MODEL ====================
def train_model(df):
    X = df[["MA5", "MA10", "Volatility", "Volume"]]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(model, "stock_model.pkl")
    return model, acc

# ============= GENAI ANALYST ====================
def generate_analysis(ticker, prob_up, prob_down, features):
    prompt = f"""
    You are a stock market analyst. Based on the ML model prediction:
    - Stock: {ticker}
    - Probability Up: {prob_up:.2f}
    - Probability Down: {prob_down:.2f}
    - Key features: {features.to_dict()}

    Write a short analyst-style explanation (3-5 sentences).
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # lightweight, fast
        contents=prompt
    )
    return response.text.strip()

# ============= STREAMLIT APP ====================
st.title("📈 Stock Price Movement Predictor + AI Analyst")

st.sidebar.header("Settings")
popular_tickers = [
    "AAPL", "TSLA", "MSFT", "AMZN", "GOOG",
    "NVDA", "META", "JPM", "BAC", "NFLX"
]
ticker = st.sidebar.selectbox("Select Stock Ticker", popular_tickers, index=0)


if st.sidebar.button("Predict"):
    df = load_data(ticker)
    if df is None:
     st.stop()  # stop here, app won’t crash


    try:
        model = joblib.load("stock_model.pkl")
    except:
        model, acc = train_model(df)
        st.write(f"Model trained. Accuracy: {acc:.2f}")

    latest_features = df[["MA5", "MA10", "Volatility", "Volume"]].iloc[-1]
    probs = model.predict_proba([latest_features])[0]
 # --- Interactive date range selection ---
    st.subheader(f"{ticker} - Select Date Range")
    start_date = st.date_input("Start date", df.index.min().date())
    end_date = st.date_input("End date", df.index.max().date())
    df_filtered = df.loc[start_date:end_date]

    # --- Closing Price + MA chart ---
    st.subheader(f"{ticker} - Closing Price & Moving Averages")
    chart_df = df_filtered[["Close", "MA5", "MA10"]]
    st.line_chart(chart_df)

    # --- Volume bar chart ---
    st.subheader(f"{ticker} - Trading Volume")
    st.bar_chart(df_filtered["Volume"])

    # --- Color-coded probability bars ---
    latest_features = df[["MA5", "MA10", "Volatility", "Volume"]].iloc[-1]
    probs = model.predict_proba([latest_features])[0]
    prob_up, prob_down = probs[1], probs[0]

    st.subheader(f"Prediction for {ticker} (next day)")
    st.write("Probability of going UP/DOWN:")
    st.progress(int(prob_up * 100))  # green-ish up bar
    st.write(f"🔼 Up: {prob_up:.2f}")
    st.write(f"🔽 Down: {prob_down:.2f}")

    prob_up, prob_down = probs[1], probs[0]
    st.subheader(f"Prediction for {ticker} (next day)")
    st.write(f"🔼 Probability Up: {prob_up:.2f}")
    st.write(f"🔽 Probability Down: {prob_down:.2f}")

    analysis = generate_analysis(ticker, prob_up, prob_down, latest_features)
    st.subheader("AI Analyst Report")
    st.write(analysis)
