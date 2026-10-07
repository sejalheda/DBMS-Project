import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("priority_dataset.csv")

# Input columns
X = data[
    [
        "description",
        "category",
        "location_type",
        "duplicate_count"
    ]
]

# Output
y = data["priority"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "description",
            TfidfVectorizer(),
            "description"
        ),
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            ["category"]
        ),
        (
            "location_type",
            OneHotEncoder(handle_unknown="ignore"),
            ["location_type"]
        ),
        (
            "duplicate_count",
            SimpleImputer(strategy="constant", fill_value=0),
            ["duplicate_count"]
        )
    ]
)


# Machine Learning model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)


# Train
model.fit(X_train, y_train)


# Test
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Save model
joblib.dump(model, "priority_model.pkl")

print("\nModel saved as priority_model.pkl")