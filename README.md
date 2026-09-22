# ⚡ Power Consumption Forecasting System

A deep learning project that predicts short-term electrical power consumption using LSTM (Long Short-Term Memory) neural networks trained on real smart grid data from Tetuan City, Morocco.

---

## 🎯 Problem Statement

Energy providers need accurate short-term load forecasting to balance supply and demand efficiently. This project builds an LSTM model that predicts the next 10-minute power consumption for Zone 1 of Tetuan City's electrical grid using historical consumption data and weather conditions.

---

## 📊 Dataset

- **Source:** Tetuan City Power Consumption Dataset (Kaggle)
- **Size:** 52,416 rows × 9 columns
- **Frequency:** Every 10 minutes
- **Period:** January 2017 – December 2017 (full year)
- **Target Column:** PowerConsumption_Zone1

---

## 🏗️ Project Architecture
Raw Data (52,416 rows, 10-min intervals)
↓
Feature Engineering
(hour, dayofweek, month, is_weekend,
Zone1_MA144, Zone1_MA1008)
↓
Chronological Train/Test Split (80/20)
↓
StandardScaler (fit on train only)
↓
Sliding Window Sequence Creation
WINDOW = 144 (24 hours of context)
↓
LSTM Model (2 layers + Dropout)
↓
Predict next 10-min Zone 1 consumption
↓
Inverse Transform → Real kW value
↓
Streamlit Web App

---

---

## 🔧 Features Used

| Feature | Type | Description |
|---|---|---|
| PowerConsumption_Zone1 | Target | Power demand in kW |
| Temperature | Weather | Ambient temperature °C |
| Humidity | Weather | Relative humidity % |
| WindSpeed | Weather | Wind speed |
| GeneralDiffuseFlows | Solar | General diffuse solar flows |
| DiffuseFlows | Solar | Diffuse solar flows |
| hour | Time | Hour of day 0-23 |
| dayofweek | Time | Day of week 0-6 |
| month | Time | Month 1-12 |
| is_weekend | Time | Weekend flag 0 or 1 |
| Zone1_MA144 | Engineered | 24-hour rolling average |
| Zone1_MA1008 | Engineered | 7-day rolling average |

---

## 🤖 Model Architecture

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(144, 12)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

**Why LSTM?**
Power consumption is a time-series problem. Consumption patterns depend heavily on recent history — daily cycles, weekly patterns, seasonal effects. LSTM memory cells capture these temporal dependencies effectively, unlike traditional ML models that treat each row independently.

**Why StandardScaler over MinMaxScaler?**
Power consumption has occasional demand spikes that would anchor MinMaxScaler's max boundary, compressing all normal readings into a narrow range near zero. StandardScaler's mean-zero unit-variance normalization handles these spikes more robustly while preserving meaningful differences between typical and high-demand periods.

**Why WINDOW = 144?**
144 x 10-minute intervals = exactly 24 hours of context. This gives the model a full day's worth of patterns before making each prediction — capturing daily usage cycles such as morning ramp-up, afternoon peak, and night-time lows.

**Why drop Zone 2 and Zone 3?**
Zone 2 and Zone 3 correlate 0.83 and 0.75 with Zone 1 respectively — high multicollinearity. The rolling average features already encode Zone 1's recent trend, making Zone 2 and Zone 3 redundant as separate inputs. Removing them reduces model complexity without losing predictive signal.

---

## 📈 Results

| Metric | LSTM Model | Naive Baseline |
|---|---|---|
| RMSE | ADD YOUR VALUE kW | ADD YOUR VALUE kW |
| MAE | ADD YOUR VALUE kW | — |
| Improvement | ADD YOUR % better than naive | — |

> Naive baseline = predicting next interval equals current interval.
> Beating this baseline by a meaningful margin confirms the LSTM is
> learning real temporal patterns, not just memorizing recent values.

> Evaluated on chronological 80/20 split — test set is the most
> recent 20% of dates, never seen during training.

---

## 🌐 Streamlit Web App

The app allows users to input current weather and power conditions and receive a prediction for the next 10-minute interval in real time.

**User Inputs:**
- Current Power Zone 1 reading (kW)
- Temperature (°C)
- Humidity (%)
- Wind Speed
- General Diffuse Flows
- Diffuse Flows
- Hour of day
- Day of week
- Month

**Output:**
- Predicted next-interval power consumption in kW
- Demand category: Low / Moderate / High / Very High
- Comparison against dataset daily average

---

## 🚀 How to Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/AtharvaPandey-coder/Power-Consumption.git
cd Power-Consumption
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**

```bash
streamlit run powerapp.py
```

---

## 📦 Requirements
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
streamlit

---

## 💡 Key Technical Decisions and Learnings

**1. Chronological train/test split — no shuffling**
Used manual index slicing to preserve time order instead of sklearn's train_test_split with shuffling. Shuffling would allow future data to leak into training, which is data leakage in time-series — the model would appear accurate but fail completely in real deployment.

**2. Fit scaler only on training data**
StandardScaler was fit exclusively on train_data, then used to transform both train and test. Fitting on the full dataset would leak test set statistics (mean and std) into training — another form of data leakage.

**3. StandardScaler chosen deliberately over MinMaxScaler**
MinMaxScaler anchors its range to the observed min and max. A single demand spike becomes the max boundary, compressing all normal readings. StandardScaler normalizes around the mean with unit variance, making it robust to spikes without distorting the distribution of typical values.

**4. Rolling averages as explicit features**
Added 24-hour (Zone1_MA144) and 7-day (Zone1_MA1008) rolling averages as input features to give the model explicit trend context. This helps the model distinguish between a reading that is high because demand is genuinely rising versus one that is high due to a one-off anomaly.

**5. Inverse transform via dummy array**
StandardScaler expects all feature columns to perform inverse transformation. A dummy array of zeros is created, the scaled prediction is inserted at the correct column index, inverse_transform is applied to the full dummy, then only the target column is extracted. This is necessary because the scaler was fitted on all features together.

---

## 🔮 Future Improvements

- Add public holiday calendar as a binary feature
- Predict all three zones simultaneously using multi-output LSTM
- Add attention mechanism on top of LSTM layers
- Experiment with Transformer-based time-series models
- Deploy on Streamlit Cloud for public access
- Integrate real-time smart meter data feed

---

## 👨‍💻 Author

**Atharva Pandey**
B.Tech Computer Science — Data Science and Machine Learning
[GitHub Profile](https://github.com/AtharvaPandey-coder)

---

## 📄 License

This project is open source and available under the MIT License.
