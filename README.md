# Drilling Fluid Report Extractor + ML

Streamlit app that extracts structured drilling fluid data from PDF reports and predicts performance degradation using machine learning.

## Features
- Upload multiple PDF reports
- Extract drilling parameters
- Predict degradation (LGS%, PV, Losses, etc.)
- Download results as CSV

## Usage
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker
```bash
docker build -t fluid-ml .
docker run -p 8501:8501 fluid-ml
```
