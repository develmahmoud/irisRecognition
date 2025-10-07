import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------- STREAMLIT APP ---------------- #

st.title("🌸 Iris Recognition using ANN + KMeans (Kaggle CSV)")

# Sidebar config
st.sidebar.header("Configuration")
epochs = st.sidebar.slider("Training Epochs", 10, 200, 50, 10)
test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2, 0.05)

# Step 1: Upload Dataset
st.subheader("Step 1: Upload Iris Dataset (CSV from Kaggle)")
uploaded_file = st.file_uploader("Upload Iris CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop Id column if exists
    if "Id" in df.columns:
        df.drop("Id", axis=1, inplace=True)

    st.write("📊 Dataset Preview:")
    st.dataframe(df.head())

    # Features & target
    X = df.drop("Species", axis=1).values
    y = df["Species"].astype("category").cat.codes
    class_names = df["Species"].astype("category").cat.categories

    # Step 2: Preprocessing
    st.subheader("Step 2: Preprocessing")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    st.write("Cluster distribution:")
    st.bar_chart(df["Cluster"].value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    # Step 3: Train ANN
    st.subheader("Step 3: Train ANN Model")

    model = Sequential([
        Dense(8, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(8, activation="relu"),
        Dense(len(class_names), activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_split=0.2, verbose=0)

    # Plot training history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history["accuracy"], label="Train Acc")
    ax[0].plot(history.history["val_accuracy"], label="Val Acc")
    ax[0].set_title("Accuracy")
    ax[0].legend()

    ax[1].plot(history.history["loss"], label="Train Loss")
    ax[1].plot(history.history["val_loss"], label="Val Loss")
    ax[1].set_title("Loss")
    ax[1].legend()
    st.pyplot(fig)

    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"✅ Test Accuracy: {acc:.2f}")

    # Step 4: Prediction
    st.subheader("Step 4: Test on New Data")
    inputs = []
    for col in df.drop(["Species", "Cluster"], axis=1).columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        inputs.append(val)

    if st.button("Predict Iris Class"):
        new_data = np.array([inputs])
        new_data_scaled = scaler.transform(new_data)
        prediction = np.argmax(model.predict(new_data_scaled), axis=-1)[0]
        st.info(f"🔮 Predicted Class: {class_names[prediction]}")

else:
    st.warning("⚠️ Please upload a Kaggle Iris CSV file to continue.")
