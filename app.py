import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crime Analysis & ANN Predictor",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Crime Analysis & ANN High-Risk Predictor")
st.markdown(
    "End-to-end pipeline: **data loading → EDA → feature engineering → ANN training → evaluation**"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    test_size    = st.slider("Test split size", 0.1, 0.4, 0.2, 0.05)
    epochs       = st.slider("Training epochs", 5, 50, 10, 5)
    batch_size   = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    run_training = st.button("🚀 Load Data & Train Model", use_container_width=True)
    st.subheader("🧠 Model Architecture")
    num_layers      = st.slider("Hidden layers", 1, 5, 2)
    neurons         = st.slider("Neurons per layer", 16, 256, 64, step=16)
    activation_fn   = st.selectbox("Activation function", ["relu", "tanh", "sigmoid", "elu", "selu"])

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1-HOFFJp86vLJle2K4k7Q5M2NCtWzdEbHQeszfOTjdaU"
    "/export?format=csv"
)

@st.cache_data(show_spinner="Downloading dataset…")
def load_raw_data():
    df = pd.read_csv(DATA_URL)
    return df

# ── Preprocessing (cached on hyperparams) ────────────────────────────────────
@st.cache_data(show_spinner="Preprocessing data…")
def preprocess(df: pd.DataFrame):
    d = df.copy()

    # Weapon Used NaN fill
    d['Weapon Used'] = d['Weapon Used'].fillna('No weapon')

    # Parse datetimes
    for col in ['Date Reported', 'Time of Occurrence']:
        d[col] = pd.to_datetime(d[col], format='mixed', dayfirst=True)
    d['Date Case Closed'] = pd.to_datetime(
        d['Date Case Closed'], format='mixed', dayfirst=True, errors='coerce'
    )

    # Split date / time components
    d['Time Reported']   = d['Date Reported'].dt.time
    d['Date Reported']   = d['Date Reported'].dt.date
    d['Date of Occurrence'] = d['Time of Occurrence'].dt.date
    d['Time of Occurrence'] = d['Time of Occurrence'].dt.time
    d['Time Case Closed']   = d['Date Case Closed'].dt.time
    d['Date Case Closed']   = d['Date Case Closed'].dt.date

    # Time features
    occ = pd.to_datetime(d['Date of Occurrence'])
    d['Year']       = occ.dt.year
    d['Month']      = occ.dt.month
    d['Day']        = occ.dt.day
    d['Day of Week']= occ.dt.dayofweek
    d['Hour']       = d['Time of Occurrence'].apply(lambda x: x.hour)

    # High-Risk target
    crime_counts = d.groupby(['City', 'Month']).size().reset_index(name='Crime Count')
    mean_count   = crime_counts['Crime Count'].mean()
    d = pd.merge(d, crime_counts, on=['City', 'Month'], how='left')
    d['High Risk'] = (d['Crime Count'] > mean_count).astype(int)

    # Encodings
    d['Victim Gender'] = d['Victim Gender'].map({'M': 0, 'F': 1}).astype(float)
    d['Case Closed']   = d['Case Closed'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # One-hot encode
    cat_cols = ['City', 'Crime Domain', 'Weapon Used', 'Crime Description']
    d = pd.get_dummies(d, columns=cat_cols, drop_first=True)

    # Standard scale
    num_cols = d.select_dtypes(include=['int64', 'float64', 'uint8', 'bool']).columns.tolist()
    exclude  = ['Report Number', 'Victim Gender', 'High Risk', 'Crime Count', 'Case Closed']
    to_scale = [c for c in num_cols if c not in exclude]
    scaler   = StandardScaler()
    d[to_scale] = scaler.fit_transform(d[to_scale])

    return d, mean_count


@st.cache_data(show_spinner="Building feature matrix…")
def build_XY(_df):
    drop_cols = [
        'Report Number', 'Date Reported', 'Time of Occurrence',
        'Date Case Closed', 'Time Reported', 'Date of Occurrence',
        'Time Case Closed', 'Crime Count', 'High Risk'
    ]
    X = _df.drop(columns=drop_cols, errors='ignore')
    for col in X.select_dtypes(include=['object']).columns:
        try:
            X[col] = pd.to_numeric(X[col])
        except ValueError:
            X = X.drop(columns=[col])
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.mean(numeric_only=True))
    y = _df['High Risk']
    return X, y


# ── ANN training (NOT cached – changes every run) ────────────────────────────

def train_model(X_train, y_train, X_test, y_test, epochs, batch_size, num_layers, neurons, activation_fn):
    input_shape = (X_train.shape[1],)
    
    model_layers = [layers.Dense(neurons, activation=activation_fn, input_shape=input_shape)]
    for _ in range(num_layers - 1):
        model_layers.append(layers.Dense(neurons, activation=activation_fn))
    model_layers.append(layers.Dense(1, activation='sigmoid'))  # output layer always sigmoid
    
    model = keras.Sequential(model_layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    progress_bar = st.progress(0, text="Training ANN…")
    epoch_log    = []
    
    class StreamlitCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pct = int((epoch + 1) / epochs * 100)
            progress_bar.progress(
                pct,
                text=f"Epoch {epoch+1}/{epochs} | "
                     f"loss={logs['loss']:.4f} | "
                     f"val_acc={logs['val_accuracy']:.4f}"
            )
            epoch_log.append(logs)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[StreamlitCallback()],
    )
    progress_bar.empty()
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════════
if run_training:

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    raw_df = load_raw_data()

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    st.header("📊 Exploratory Data Analysis")

    with st.expander("📄 Raw Data Preview", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True)
        st.caption(f"Shape: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

    col1, col2 = st.columns(2)

    # Crime by city
    with col1:
        st.subheader("Top 10 Cities by Crime Count")
        top10 = raw_df['City'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top10.index, y=top10.values, palette='viridis', ax=ax)
        ax.set_xlabel("City"); ax.set_ylabel("Number of Crimes")
        ax.set_title("Top 10 Cities by Crime Count")
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Crime domain distribution
    with col2:
        st.subheader("Crime Domain Distribution")
        domain_counts = raw_df['Crime Domain'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=domain_counts.index, y=domain_counts.values, palette='magma', ax=ax)
        ax.set_xlabel("Crime Domain"); ax.set_ylabel("Number of Crimes")
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    # Weapon usage pie
    with col3:
        st.subheader("Weapon Usage Distribution")
        weapon_counts = raw_df['Weapon Used'].fillna('No weapon').value_counts()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(weapon_counts, labels=weapon_counts.index, autopct='%1.1f%%',
               startangle=140, pctdistance=0.85)
        ax.set_title("Distribution of Weapon Used"); ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Victim gender donut
    with col4:
        st.subheader("Victim Gender Distribution")
        gender_counts = raw_df['Victim Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
               startangle=140, pctdistance=0.85, wedgeprops=dict(width=0.4))
        ax.set_title("Victim Gender (Donut)"); ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Crimes over time
    st.subheader("Crimes Over Time (Monthly)")
    tmp = raw_df.copy()
    tmp['Date of Occurrence'] = pd.to_datetime(
        tmp['Time of Occurrence'], format='mixed', dayfirst=True, errors='coerce'
    )
    tmp['Month_Period'] = tmp['Date of Occurrence'].dt.to_period('M')
    monthly = tmp.groupby('Month_Period').size().reset_index(name='Crime Count')
    monthly['Month_Period'] = monthly['Month_Period'].astype(str)
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.lineplot(x='Month_Period', y='Crime Count', data=monthly, marker='o', ax=ax)
    ax.set_xlabel("Month"); ax.set_ylabel("Number of Crimes")
    ax.set_title("Crimes Over Time (Month-wise)")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=90); plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    st.header("🛠️ Feature Engineering & Preprocessing")
    with st.spinner("Processing…"):
        processed_df, mean_crime_count = preprocess(raw_df)
        X, y = build_XY(processed_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{len(X):,}")
    c2.metric("Feature Count", f"{X.shape[1]}")
    c3.metric("High-Risk Threshold (mean crimes/city-month)", f"{mean_crime_count:.1f}")

    hr_dist = y.value_counts(normalize=True) * 100
    st.info(
        f"**Class distribution** — High Risk (1): {hr_dist.get(1, 0):.1f}%  |  "
        f"Low Risk (0): {hr_dist.get(0, 0):.1f}%"
    )

    # ── 4. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    st.write(
        f"Train: **{len(X_train):,}** samples · Test: **{len(X_test):,}** samples"
    )

    st.divider()

    # ── 5. Model training ─────────────────────────────────────────────────────
    st.header("🧠 ANN Model Training")
    model, history = train_model(
    X_train, y_train, X_test, y_test,
    epochs, batch_size, num_layers, neurons, activation_fn
    )

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    st.header("📈 Model Evaluation")

    y_pred_proba = model.predict(X_test).ravel()
    y_pred       = (y_pred_proba > 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_proba)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall",    f"{rec:.4f}")
    m4.metric("F1 Score",  f"{f1:.4f}")
    m5.metric("AUC",       f"{auc:.4f}")

    col_cm, col_roc = st.columns(2)

    # Confusion matrix
    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ROC curve
    with col_roc:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='steelblue', label=f'AUC = {auc:.4f}')
        ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Classification report
    with st.expander("📋 Full Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.success("✅ Pipeline complete!")

else:
    st.info(
        "👈 Adjust settings in the sidebar then click **Load Data & Train Model** to begin."
    )
    st.markdown("""
    ### What this app does
    1. **Loads** crime data from a Google Sheets CSV export  
    2. **Explores** the data with interactive charts (cities, domains, weapons, time trends)  
    3. **Engineers features** — time components, High-Risk label, one-hot encoding, scaling  
    4. **Trains** a 3-layer ANN (Dense 64 → 32 → 1 sigmoid) with configurable hyperparameters  
    5. **Evaluates** with Accuracy, Precision, Recall, F1, AUC, Confusion Matrix, and ROC Curve  
    """)
