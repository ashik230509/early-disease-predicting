# Early Disease Predicting

A machine learning project for predicting early disease indicators using heart disease data.

## Project Structure

- `app.py` - Streamlit web application
- `train_models.py` - Model training and evaluation scripts
- `heart.csv` - Dataset for training
- `requirements.txt` - Python dependencies
- `outputs/` - Generated models, plots, and results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train models:
```bash
python train_models.py
```

3. Run the web app:
```bash
streamlit run app.py
```

## Requirements

See `requirements.txt` for all dependencies including:
- TensorFlow/Keras
- Scikit-learn
- Pandas
- Streamlit
- Matplotlib & Seaborn
