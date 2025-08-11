
---

# Ethereum Price Prediction using LSTM

## Overview
This project focuses more on predicting the future prices of Ethereum (ETH-USD) using historical financial data and deep learning's long-short term memory techniques. The project leverages various tools for data preprocessing, scaling, and applying deep learning models like Long Short-Term Memory (LSTM) to forecast Ethereum prices.

## Project Structure
- `ETH.ipynb`: The main Jupyter notebook where Ethereum price prediction models are built & evaluated.
- `data/`: Directory where the historical ETH-USD price data is stored (you may need to fetch or provide your own dataset).
- `README.md`: Overview and instructions for the project.

## Features
- **Historical Price Data**: Historical prices of Ethereum (ETH-USD) from external sources.
- **Preprocessing**: Data scaling using MinMaxScaler and feature engineering for model training.
- **Deep Learning Model**: An LSTM neural network is used to predict Ethereum prices based on historical data.
- **Visualizations**: Plots showing price trends and predictions vs. actual prices.

## Requirements
To access and run  this project locally, you'll need the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tensorflow`
This requirement can be automatically installed by running the  command line below

You can install these dependencies using:
```bash
pip install -r requirements.txt
```

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/Ibidapo111/Stock-Price-Prediction-ETH.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Stock-Price-Prediction
   ```
3. Open the `ETH.ipynb` notebook to view the code and run the prediction model.

## Data
- The data used for this project is Ethereum's historical price data. Ensure you have the `ETH-USD.csv` file stored in the `data/` folder or download it from [Yahoo Finance](https://finance.yahoo.com/quote/ETH-USD/history/).
- The dataset should contain columns such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.

## Usage
1. **Preprocessing**: The dataset is preprocessed by normalizing the features using MinMaxScaler from Scikit-learn.
2. **Model Training**: The LSTM model from TensorFlow is trained on the preprocessed data to predict future Ethereum prices.
3. **Visualization**: After training, the model's predictions are compared with actual prices, and the results are visualized.

## Results
- The LSTM model provides future price predictions for Ethereum based on historical data.
- Visual comparisons of predicted vs. actual prices are provided.

## Future Work
- Explore more advanced forecasting techniques such as ARIMA, Prophet, or hybrid models.
- Extend the model to predict other cryptocurrencies or financial instruments.
- Integrate real-time data fetching and prediction.

## Contributing
Feel free to contribute to this project by forking the repository and submitting pull requests.

## License
This project is licensed under the MIT License.

---
