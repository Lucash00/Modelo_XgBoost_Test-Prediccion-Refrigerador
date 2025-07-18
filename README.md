# XGBoost Model for Refrigerator Failure Prediction

This project is a simple example of a machine learning model using XGBoost to predict refrigerator failures based on temperature and humidity.

## Structure

- `main.py`: main code that creates, trains, evaluates, and saves the model.
- `model/`: folder where the trained model and auxiliary data are stored.
- `requirements.txt`: required dependencies.

## Installation

Clone the repository:

```bash
git clone https://github.com/Lucash00/Modelo_XgBoost_Test-Prediccion-Refrigerador.git
cd Modelo_XgBoost_Test-Prediccion-Refrigerador
```

## Create a virtual environment and install dependencies:

```bash
python -m venv .venv

source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows

pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

This will train (or load) the model, evaluate its performance, and make a sample prediction.

## Notes

The model saves detected errors to improve future runs.

A history of metrics is stored in model/historical_metrics.csv and of actual failures in model/actual_failures.csv.