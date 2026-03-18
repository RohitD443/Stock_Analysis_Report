# ==============================
# 📦 1. IMPORT LIBRARIES
# ==============================
# We import only what we need to keep code clean and readable

import yfinance as yf              # Fetch stock data
import pandas as pd               # Data manipulation
import numpy as np                # Numerical operations
import matplotlib.pyplot as plt   # Visualization

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ==============================
# 📥 2. DATA COLLECTION
# ==============================
def load_data(ticker, start, end):
    """
    Fetch stock data from Yahoo Finance
    WHY: Keeping it in a function makes it reusable for any stock
    """
    df = yf.download(ticker, start=start, end=end)
    
    # Fix multi-level column issue (happens sometimes with yfinance)
    df.columns = df.columns.get_level_values(0)
    
    df.dropna(inplace=True)  # Remove missing values
    df.reset_index(inplace=True)  # Convert index to column
    
    return df

# ==============================
# ⚙️ 3. FEATURE ENGINEERING
# ==============================
def add_features(df):
    """
    Create technical indicators
    WHY: Models don't understand raw price → we convert into patterns
    """

    # Moving Averages (Trend indicators)
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    # Returns (used for volatility)
    df['Returns'] = df['Close'].pct_change()

    # RSI (Momentum indicator)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volatility (risk measure)
    df['Volatility'] = df['Returns'].rolling(10).std()

    # Momentum (price strength)
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # EMA (faster trend capture than MA)
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # 52 Week High & Low (important trading zones)
    df['52W_High'] = df['Close'].rolling(252).max()
    df['52W_Low'] = df['Close'].rolling(252).min()

    df.dropna(inplace=True)

    return df

# ==============================
# 🎯 4. TARGET CREATION
# ==============================
def create_target(df):
    """
    Predict next day's price
    WHY: Makes it a supervised learning problem
    """
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

# ==============================
# ✂️ 5. TRAIN-TEST SPLIT
# ==============================
def split_data(df):
    """
    Time-based split (NOT random)
    WHY: Stock data is sequential → random split causes leakage
    """
    features = ['Close', 'MA50', 'MA200', 'RSI', 'Volatility', 'Momentum', 'EMA50']
    
    X = df[features]
    y = df['Target']

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test

# ==============================
# 🤖 6. MODEL TRAINING
# ==============================
def train_models(X_train, y_train):
    """
    Train multiple models for comparison
    WHY: No single model works best always → compare performance
    """

    # Linear Regression (baseline model)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Random Forest (handles non-linearity well)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # XGBoost (best for structured/tabular data)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4)
    xgb.fit(X_train, y_train)

    return lr, rf, xgb

# ==============================
# 📊 7. MODEL EVALUATION
# ==============================
def evaluate_model(name, y_true, y_pred):
    """
    Evaluate using RMSE
    WHY: RMSE gives error in same unit as price → easy to interpret
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} RMSE: {rmse:.2f}")

# ==============================
# 📉 8. SIGNAL GENERATION
# ==============================
def generate_signals(df):
    """
    Create Buy/Sell signals
    WHY: Converts analysis into actionable strategy
    """

    df['Buy_Signal'] = np.where(
        (df['Close'] <= df['52W_Low'] * 1.05) & (df['RSI'] < 30),
        1, 0
    )

    df['Sell_Signal'] = np.where(
        (df['Close'] >= df['52W_High'] * 0.95) & (df['RSI'] > 70),
        1, 0
    )

    return df

# ==============================
# 📊 9. VISUALIZATION
# ==============================
def plot_results(df, y_test, preds):
    """
    Plot predictions vs actual
    WHY: Visual comparison helps understand model performance better
    """

    plt.figure(figsize=(12,6))
    plt.plot(y_test.values, label="Actual")

    for name, pred in preds.items():
        plt.plot(pred, label=name)

    plt.legend()
    plt.title("Model Comparison")
    plt.show()


def plot_signals(df):
    """
    Plot Buy/Sell signals
    """

    buy = df[df['Buy_Signal'] == 1]
    sell = df[df['Sell_Signal'] == 1]

    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Price', color='black')

    plt.scatter(buy['Date'], buy['Close'], color='green', marker='^', s=100)
    plt.scatter(sell['Date'], sell['Close'], color='red', marker='v', s=100)

    plt.title("Trading Signals")
    plt.legend()
    plt.show()

# ==============================
# 🚀 10. MAIN EXECUTION PIPELINE
# ==============================
if __name__ == "__main__":

    # Step 1: Load data
    stock = load_data("RELIANCE.NS", "2018-01-01", "2025-12-31")

    # Step 2: Feature engineering
    stock = add_features(stock)

    # Step 3: Create target
    stock = create_target(stock)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(stock)

    # Step 5: Train models
    lr, rf, xgb = train_models(X_train, y_train)

    # Step 6: Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    # Step 7: Evaluation
    evaluate_model("Linear Regression", y_test, lr_pred)
    evaluate_model("Random Forest", y_test, rf_pred)
    evaluate_model("XGBoost", y_test, xgb_pred)

    # Step 8: Plot comparison
    plot_results(
        stock,
        y_test,
        {"Linear": lr_pred, "RF": rf_pred, "XGB": xgb_pred}
    )

    # Step 9: Generate signals
    stock = generate_signals(stock)

    # Step 10: Plot signals
    plot_signals(stock)
