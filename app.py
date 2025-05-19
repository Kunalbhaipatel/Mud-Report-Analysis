def to_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def simulate_label(row):
    return int(
        to_float(row['LGS%']) > 10 or
        to_float(row['Losses']) > 200 or
        to_float(row['PV']) > 35
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
