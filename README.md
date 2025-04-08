# Bitcoin Risk Management Using Binance API

This project is a **Bitcoin price prediction and risk management application** that leverages Binance API data, machine learning models, and Streamlit for visualization. The app processes historical Bitcoin data, generates features, and provides predictions using various models such as Linear Regression, LSTM, and Stochastic Simulation. 

App link: https://bitcoinpricepred.streamlit.app/

---

## **Features**
- **Data Fetching**: Automatically fetches historical Bitcoin data from Binance API every 4 hours.
- **Data Processing**: Processes raw data into features for machine learning models.
- **Prediction Models**:
  - **Linear Regression**: Predicts the next day's closing price.
  - **LSTM**: Predicts the next day's closing price using sequential data.
  - **Stochastic Simulation**: Simulates future Bitcoin price movements and calculates probabilities of significant price changes.
- **Interactive Visualization**: Displays predictions and metrics in an interactive Streamlit app.

---

## **Project Structure**

```
bitcoinPred/
├── app.py                     # Main Streamlit app for predictions and visualization
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── src/                       # Source code directory
│   ├── data/
│   │   ├── process_data.py            # Processes raw data into features
│   │   ├── process_linear_next.py     # Processes data for Linear Regression model
│   │   ├── process_linear_next_live.py # Processes live data for Linear Regression
│   │   ├── process_lstm.py            # Processes data for LSTM model
│   │   ├── process_lstm_live.py       # Processes live data for LSTM model
│   ├── models/
│   │   ├── stochastic_sim.py          # Stochastic simulation for price prediction
│   │   ├── linear_regression_next.pkl # Trained Linear Regression model
│   │   ├── lstm_daily_model.keras     # Trained LSTM model
├── data/
│   ├── raw/                           # Raw data fetched from Binance API
│   ├── processed/                     # Processed data ready for modeling
└── .env                               # Environment variables (API keys, AWS credentials)
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/bitcoinPred.git
cd bitcoinPred
```

### **2. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **3. Set Up Environment Variables**
Create a `.env` file in the root directory with the following content:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

### **4. Run the App**
Start the Streamlit app:
```bash
streamlit run app.py
```

---

## **How It Works**

### **1. Data Fetching**
The `save_data.py` script fetches historical Bitcoin data from Binance API and uploads it to an AWS S3 bucket every 4 hours (via a cron job).

### **2. Data Processing**
The following scripts process the raw data into features for machine learning models:
- `process_data.py`: General feature engineering.
- `process_linear_next.py` and `process_linear_next_live.py`: Feature engineering for Linear Regression.
- `process_lstm.py` and `process_lstm_live.py`: Feature engineering for LSTM.

### **3. Prediction Models**
- **Linear Regression**:
  - Predicts the next day's closing price using engineered features.
- **LSTM**:
  - Uses sequential data to predict the next day's closing price.
- **Stochastic Simulation**:
  - Simulates future price movements and calculates probabilities of significant price changes using Geometric Browninan Motion.

### **4. Visualization**
The Streamlit app provides an interactive interface to:
- View predictions and metrics.
- Visualize actual vs. predicted prices.
- Run stochastic simulations.

---

## **Key Features in the App**
- **Automatic Data Processing**:
  - Automatically processes data when the app starts.
- **Manual Data Processing**:
  - Includes a "Process Data" button to manually re-trigger data processing.
- **Model Selection**:
  - Choose between Linear Regression, LSTM, and Stochastic Simulation.
- **Interactive Plots**:
  - Visualize predictions and simulation results.

---

## **Deployment**
The app can be deployed using Streamlit Community Cloud or any other hosting platform. To deploy on Streamlit Community Cloud:
1. Push the project to GitHub.
2. Go to Streamlit Community Cloud.
3. Select the repository and deploy the app.

---

## **Technologies Used**
- **Python**: Core programming language.
- **Streamlit**: Interactive web app framework.
- **Binance API**: Fetches historical Bitcoin data.
- **AWS S3**: Stores raw and processed data.
- **AWS EC2**: Fetches live data from binance automatically.
- **Matplotlib**: Data visualization.
- **Scikit-learn**: Machine learning models and preprocessing.
- **TensorFlow/Keras**: LSTM model for sequential data.

---

## **Future Improvements**
- Add support for additional cryptocurrencies.
- Decrease fetching window from 4 hours.
- Enhance the UI with more interactive visualizations.
- Add advanced risk management metrics.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request.

---

## **License**
This project is licensed under the MIT License.