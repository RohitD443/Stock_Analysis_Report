# Stock_Analysis_Report
🚀 Overview

This project focuses on analyzing stock market data, predicting future prices using machine learning models, and generating actionable Buy/Sell signals based on technical indicators.

The goal is to combine data analysis + machine learning + trading logic into a single end-to-end pipeline.

🎯 Objectives

Analyze historical stock price data

Engineer meaningful financial features

Predict next-day stock prices

Compare multiple ML models

Generate trading signals (Buy/Sell)

Visualize model performance and strategy
📦 Technologies Used

Python

Pandas / NumPy → Data processing

Matplotlib → Visualization

yFinance → Stock data collection

Scikit-learn → ML models

XGBoost → Advanced boosting model

📥 Data Collection

Stock data is fetched using Yahoo Finance:

yf.download("RELIANCE.NS", start="2018-01-01", end="2025-12-31")
Why?

Reliable historical data

Easy API integration

Covers Indian stock market

⚙️ Feature Engineering

We create multiple technical indicators to help models understand market behavior:

Feature	Purpose
MA50 / MA200	Trend detection
RSI	Overbought / Oversold
Volatility	Risk measurement
Momentum	Price strength
EMA50	Faster trend indicator
Returns	Daily price change
52W High/Low	Key price zones
Why Feature Engineering?

Raw prices don’t reveal patterns clearly. These indicators help models learn:

Trends

Market sentiment

Risk levels

🎯 Target Variable
df['Target'] = df['Close'].shift(-1)
Meaning:

We predict next day’s closing price

Why?

Converts problem into supervised learning

Useful for short-term trading strategies

✂️ Train-Test Split
split = int(len(df) * 0.8)

80% → Training data

20% → Testing data

Important:

This is a time-based split, not random.

Why?

Stock data is sequential:
👉 Past → Present → Future
Random splitting would cause data leakage

🤖 Models Used
1. Linear Regression

Simple baseline model

Assumes linear relationship

2. Random Forest

Captures non-linear patterns

Reduces overfitting

3. XGBoost

Advanced boosting algorithm

High performance on structured data

📊 Evaluation Metric
RMSE = √(Mean Squared Error)
Why RMSE?

Easy to interpret (same unit as price)

Penalizes large errors more

📉 Trading Strategy
🟢 Buy Signal
(Close <= 52W Low * 1.05) AND (RSI < 30)

👉 Buy when:

Price is near yearly low (cheap)

RSI shows oversold condition

🔴 Sell Signal
(Close >= 52W High * 0.95) AND (RSI > 70)

👉 Sell when:

Price is near yearly high (expensive)

RSI shows overbought condition

📊 Visualization
1. Price vs Moving Averages

Shows trend direction

2. Model Comparison Plot

Compares actual vs predicted prices

3. Buy/Sell Signal Plot

Highlights entry and exit points

🧠 Key Learnings

Importance of time-series data handling

Feature engineering improves model performance significantly

No single model is best → comparison is essential

Combining ML + trading logic creates real-world value

⚠️ Limitations

Does not account for:

News / macro events

Market sentiment

Sudden price shocks

Purely based on historical data

🔮 Future Improvements

Add LSTM / Deep Learning models

Implement Backtesting (profit calculation)

Build Streamlit dashboard

Add classification model (Up/Down prediction)

Include fundamental + sentiment analysis

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run the script
python stock_analysis.py
📌 Use Case

This project is useful for:

Data Analyst / Data Scientist portfolios

Financial analytics roles

Algorithmic trading beginners

👨‍💻 Author

Rohit Deshawal

Junior Data Analyst

Skilled in SQL, Python, Power BI, Machine Learning

⭐ Final Note

This project demonstrates:

End-to-end data pipeline

Machine learning application in finance

Practical trading strategy design
