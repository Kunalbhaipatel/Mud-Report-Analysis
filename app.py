import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_info_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    data = {}
    data['Operator'] = re.search(r"Operator\s+(.*)", text).group(1).strip()
    data['Rig Name'] = re.search(r"Rig Name\s+(.*)", text).group(1).strip()
    data['Well Name'] = re.search(r"Well Name\s+(.*)", text).group(1).strip()
    data['Date'] = re.search(r"Date\s+(\d{4}-\d{2}-\d{2})", text).group(1).strip()
    data['Bit Size'] = re.search(r"Bit Size\s+([\d.]+)", text).group(1).strip()
    data['Drilling Hrs'] = re.search(r"Drilling\s+(\d+)", text).group(1).strip()
    data['Total Circ'] = re.search(r"Total Circ\s+([\d.]+)", text).group(1).strip()
    data['LGS%'] = re.search(r"LGS\s*/\s*HGS\s*%\s*([\d.]+)", text).group(1).strip()
    data['Base Oil'] = re.search(r"Base\s+([\d.]+)", text).group(1).strip()
    data['Water'] = re.search(r"Drill Water\s+([\d.]+)", text).group(1).strip()
    data['Barite'] = re.search(r"Barite\s+([\d.]+)", text).group(1).strip()
    data['Chemical'] = re.search(r"Chemicals\s+([\d.]+)", text).group(1).strip()
    data['Reserve'] = re.search(r"Reserve\s+\*\s+([\d.]+)", text).group(1).strip()
    loss_match = re.search(r"SCE\s+([\d.]+).*?Other\s+([\d.]+)", text, re.DOTALL)
    if loss_match:
        data['Losses'] = float(loss_match.group(1)) + float(loss_match.group(2))
    else:
        data['Losses'] = 0
    data['Mud Flow'] = re.search(r"gpm\s+([\d.]+)", text).group(1).strip()
    data['PV'] = re.search(r"PV\s+@.*?([\d.]+)", text).group(1).strip()
    data['YP'] = re.search(r"YP\s+lb/100ft²\s+([\d.]+)", text).group(1).strip()
    data['Mud Weight'] = re.search(r"Density\s+@.*?([\d.]+\s*@\s*[\d.]+)", text).group(1).strip()

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
