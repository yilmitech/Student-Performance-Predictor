# Academic Performance Predictor Web App

## Project Overview

This web application predicts a student's academic performance based on input features such as hours studied, attendance rate, and previous scores. It uses a trained PyTorch neural network model and a scikit-learn scaler to preprocess inputs and provide a prediction indicating whether the student is likely to pass or fail.

## Features

- User-friendly web interface to input student features.
- Backend powered by Flask serving the model prediction.
- Scaled input features with consistent preprocessing.
- Binary classification output with confidence percentage.
- Robust input validation and error handling.

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

1. Clone or download this repository.

2. Navigate to the project root directory:

    cd your-project-directory

3. Create a virtual environment (recommended):

    python3 -m venv venv

4. Activate the virtual environment:

- On Windows (CMD):

      venv\Scripts\activate

- On macOS/Linux:

      source venv/bin/activate

5. Install required dependencies:

    pip install -r requirements.txt

## Environment Setup

1. Copy the example environment file and configure it:

    cp .env.example .env

2. Edit `.env` to set appropriate values:

- `FLASK_ENV`: Set to `development` for debug mode or `production` for deployment.
- `MODEL_PATH`: Path to your PyTorch model file (default: `model.pth`).
- `SCALER_PATH`: Path to your scaler object file (default: `scaler.pkl`).
- `SECRET_KEY`: Flask secret key for session security (set to a secure random value).

## Model and Scaler Files

You need to place your trained PyTorch model and scaler object in the project root directory (or update paths in `.env` accordingly):

- `model.pth`: Serialized PyTorch model file.
- `scaler.pkl`: Serialized scikit-learn scaler object.

**Note:** These files are not included in the repository due to size and privacy. Ensure you have these files from your training pipeline.

## Running the Application

To start the Flask development server, run:

    python app.py

The app will be accessible at `http://localhost:5000`.

## Using the Web Interface

1. Open your web browser and navigate to:

       http://localhost:5000

2. Fill in the form with the student's features:

    - Hours Studied (per week)
    - Attendance Rate (%)
    - Previous Scores (average %)

3. Click the **Predict** button.

4. View the predicted academic performance on the results page.

## Testing

To run the unit tests:

    python -m unittest discover -s tests

## Troubleshooting

- **Model or scaler not loaded:**  
  Ensure `model.pth` and `scaler.pkl` are correctly placed and paths in `.env` are accurate.

- **Invalid input errors:**  
  Confirm all form inputs are numeric and within valid ranges.

- **Port conflicts:**  
  If port 5000 is in use, stop the conflicting service or modify the port in `app.py`.

- **Environment variables not loading:**  
  Install `python-dotenv` if not installed or manually export variables in your shell.

## Project Structure
