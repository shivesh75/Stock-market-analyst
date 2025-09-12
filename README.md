

# 📈 Stock Price Movement Predictor + AI Analyst

This project combines **Machine Learning** and **Generative AI** to predict stock price movement (up or down) and generate an **analyst-style explanation** of the prediction. The interactive dashboard is built with **Streamlit** for an intuitive user experience.

## 🚀 Features

* Fetches **5 years of stock data** via `yfinance`
* Computes features like **moving averages** and **volatility**
* Trains a **Random Forest model** to predict next-day stock movement
* Generates an **AI-powered analyst report** using **Google Gemini** (with optional **Ollama** integration)
* Clean, interactive dashboard to explore predictions for popular tickers (AAPL, TSLA, MSFT, etc.)

## 🛠️ Tech Stack

* **Python**
* **Machine Learning** → scikit-learn (Random Forest)
* **Data** → yfinance, pandas, numpy
* **GenAI** → Gemini API / Ollama (local LLM)
* **Frontend** → Streamlit

## 📂 Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/stock-ai-analyst.git
   cd stock-ai-analyst
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Set your Gemini API key (temporary):

   ```bash
   $env:GEMINI_API_KEY="your_api_key_here"   # PowerShell
   export GEMINI_API_KEY="your_api_key_here" # Linux/Mac
   ```
4. Run the app:

   ```bash
   streamlit run app.py
   ```

## 📊 Example

* Input: **AAPL (Apple Inc.)**
* Output:

  * **Prediction**: 65% chance stock goes up tomorrow
  * **AI Analyst Report**: *“Apple’s short-term trend shows momentum supported by stable volatility. The 5-day moving average is above the 10-day average, suggesting positive sentiment. However, trading volume has softened, which could limit upside potential.”*


