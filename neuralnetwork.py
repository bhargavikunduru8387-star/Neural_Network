import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

LEARNING_RATE = 0.001
EPOCHS = 300
BATCH_SIZE = 8

HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 8

ACTIVATION_FUNCTION = "relu"
OUTPUT_ACTIVATION = "softmax"
OPTIMIZER_NAME = "adam"
DROPOUT_RATE = 0.10

df = pd.read_csv("data/iris_dataset.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Input features and target
X = df.drop(columns=["species"]).to_numpy()

encoder = LabelEncoder()
y = encoder.fit_transform(df["species"])

print("\nFlower classes:", encoder.classes_)
print("Total samples:", len(X))

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


model = Sequential([
    tf.keras.Input(shape=(4,)),

    Dense(
        HIDDEN_LAYER_1,
        activation=ACTIVATION_FUNCTION
    ),

    Dropout(DROPOUT_RATE),

    Dense(
        HIDDEN_LAYER_2,
        activation=ACTIVATION_FUNCTION
    ),

    Dense(
        3,
        activation=OUTPUT_ACTIVATION
    )
])


optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nNeural Network Architecture:")
model.summary()


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

mlflow.set_experiment("Iris_Neural_Network")


with mlflow.start_run():
    # Log training parameters
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("hidden_layer_1", HIDDEN_LAYER_1)
    mlflow.log_param("hidden_layer_2", HIDDEN_LAYER_2)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)

    print("\nStarting model training...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\nTraining completed!")
    print("Epochs completed:", len(history.history["loss"]))

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n======================================")
    print("         FINAL MODEL RESULTS")
    print("======================================")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print("======================================")

    # Log evaluation metrics to MLflow
    mlflow.log_metric("test_loss", float(test_loss))
    mlflow.log_metric("test_accuracy", float(test_accuracy))
    mlflow.log_metric("accuracy_score", float(accuracy))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=encoder.classes_,
        zero_division=0
    ))

    new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
    new_flower_scaled = scaler.transform(new_flower)
    prediction = model.predict(new_flower_scaled, verbose=0)

    predicted_index = np.argmax(prediction[0])
    predicted_species = encoder.classes_[predicted_index]

    print("\nNew Flower Prediction:")
    print("Predicted species:", predicted_species)

    print("\nClass probabilities:")
    for name, prob in zip(encoder.classes_, prediction[0]):
        print(f"{name}: {prob * 100:.2f}%")

    from pathlib import Path

    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_DIR / "iris_neural_network.keras")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

    print("Model and preprocessing files saved successfully!")
