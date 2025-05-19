import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def safe_search(pattern, text):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

def extract_info_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    data = {}
    data['Operator'] = safe_search(r"Operator\s+(.*)", text)
    data['Rig Name'] = safe_search(r"Rig Name\s+(.*)", text)
    data['Well Name'] = safe_search(r"Well Name\s+(.*)", text)
    data['Date'] = safe_search(r"Date\s+(\d{4}-\d{2}-\d{2})", text)
    data['Bit Size'] = safe_search(r"Bit Size\s+([\d.]+)", text)
    data['Drilling Hrs'] = safe_search(r"Drilling\s+(\d+)", text)
    data['Total Circ'] = safe_search(r"Total Circ\s+([\d.]+)", text)
    data['LGS%'] = safe_search(r"LGS\s*/\s*HGS\s*%\s*([\d.]+)", text)
    data['Base Oil'] = safe_search(r"Base\s+([\d.]+)", text)
    data['Water'] = safe_search(r"Drill Water\s+([\d.]+)", text)
    data['Barite'] = safe_search(r"Barite\s+([\d.]+)", text)
    data['Chemical'] = safe_search(r"Chemicals\s+([\d.]+)", text)
    data['Reserve'] = safe_search(r"Reserve\s+\*\s+([\d.]+)", text)

    loss_match = re.search(r"SCE\s+([\d.]+).*?Other\s+([\d.]+)", text, re.DOTALL)
    if loss_match:
        data['Losses'] = float(loss_match.group(1)) + float(loss_match.group(2))
    else:
        data['Losses'] = 0

    data['Mud Flow'] = safe_search(r"gpm\s+([\d.]+)", text)
    data['PV'] = safe_search(r"PV\s+@.*?([\d.]+)", text)
    data['YP'] = safe_search(r"YP\s+lb/100ft²\s+([\d.]+)", text)
    data['Mud Weight'] = safe_search(r"Density\s+@.*?([\d.]+\s*@\s*[\d.]+)", text)

    return data
def simulate_label(row):
    return int(
        float(row['LGS%']) > 10 or
        float(row['Losses']) > 200 or
        float(row['PV']) > 35
    )

st.title("📄 Drilling Fluid Report Extractor + ML Degradation Predictor")

uploaded_files = st.file_uploader("Upload Daily Drilling Fluid PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    records = []
    for file in uploaded_files:
        try:
            record = extract_info_from_pdf(file)
            records.append(record)
        except Exception as e:
            st.error(f"Failed to parse {file.name}: {e}")

    if records:
        df = pd.DataFrame(records)
        st.success("✅ Data Extracted!")
        st.dataframe(df)

        # Simulate degradation labels
        df['Degraded'] = df.apply(simulate_label, axis=1)

        # Train ML model
        features = ['LGS%', 'PV', 'YP', 'Mud Flow', 'Losses']
        X = df[features].astype(float)
        y = df['Degraded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        df['Predicted Degradation'] = clf.predict(X)

        st.subheader("🧠 ML-Based Degradation Prediction")
        st.dataframe(df[['Date', 'LGS%', 'PV', 'YP', 'Losses', 'Predicted Degradation']])

        st.text("📊 Model Performance on Holdout Data")
        y_pred = clf.predict(X_test)
        st.text(classification_report(y_test, y_pred))

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "fluid_reports_ml.csv", "text/csv")
