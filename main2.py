import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Iris Recognition: ANN vs KMeans", layout="wide")
st.title("🌸 Iris Recognition using ANN & KMeans Clustering")
st.write(
    "Note: This app is a final year project developed under the supervision of Mal Ibrahim Madigawa at the"
    " **Department of Software Engineering, Federal University Dutse.** By "
    "**Auwal Usman Muhammad with Registration number FCP/CSE/18/1024**")
st.write(
    "This app demonstrates **Iris dataset classification with ANN** and "
    "**unsupervised clustering with KMeans** side by side. ")

# -----------------------------
# Load Dataset
# -----------------------------
st.header("1. Load Dataset")

uploaded_file = st.file_uploader("Upload your Iris CSV file (with species column)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded dataset.")
    st.dataframe(df.head())

    # Separate features and labels
    if "Species" in df.columns:
        X = df.drop(columns=["Species"]).values
        y = pd.factorize(df["Species"])[0]  # Encode species labels numerically
        species_names = df["Species"].unique()
    else:
        st.error("❌ Uploaded file must contain a 'species' column.")
        st.stop()

else:
    st.info("ℹ️ No file uploaded. Using default sklearn Iris dataset.")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [iris.target_names[i] for i in y]
    species_names = iris.target_names
    st.dataframe(df.head())

# -----------------------------
# Preprocessing & KMeans Clustering
# -----------------------------
st.header("2. KMeans Clustering")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

st.write("KMeans clustering completed. Visualizations below:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Clusters (KMeans)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df.iloc[:, 2], df.iloc[:, 3],  # Petal length vs Petal width (assumed)
        c=df["cluster"], cmap="viridis", s=50
    )
    ax.set_xlabel(df.columns[2])
    ax.set_ylabel(df.columns[3])
    st.pyplot(fig)

with col2:
    st.subheader("True Species")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df.iloc[:, 2], df.iloc[:, 3],
        c=y, cmap="viridis", s=50
    )
    ax.set_xlabel(df.columns[2])
    ax.set_ylabel(df.columns[3])
    st.pyplot(fig)

st.subheader("Cluster vs Species Comparison")
comparison = pd.crosstab(df["cluster"], df["Species"])
st.dataframe(comparison)

# -----------------------------
# ANN Classification
# -----------------------------
st.header("3. Artificial Neural Network (ANN) Classification")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build ANN model
model = Sequential([
    Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(len(np.unique(y)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0, validation_split=0.2)

# Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
ann_accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ ANN Classification Accuracy on Test Data: {ann_accuracy:.2f}")

# -----------------------------
# Accuracy Visualization
# -----------------------------
st.header("4. ANN Training Progress")

fig, ax = plt.subplots()
ax.plot(history.history["accuracy"], label="Train Accuracy")
ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.legend()
ax.set_title("ANN Training Progress")
st.pyplot(fig)

# -----------------------------
# Side by Side Performance Comparison
# -----------------------------
st.header("5. ANN vs KMeans Comparison")

col1, col2 = st.columns(2)

with col1:
    st.metric("ANN Test Accuracy", f"{ann_accuracy:.2f}")

with col2:
    inertia = kmeans.inertia_
    st.metric("KMeans Inertia", f"{inertia:.2f}")

st.info(
    "✅ ANN is supervised (uses true labels) while KMeans is unsupervised. Use the comparison above to evaluate performance differences.")
