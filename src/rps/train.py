from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "samples.csv"
MODEL_OUT = PROJECT_ROOT / "data" / "models" / "rps_model.joblib"


LABEL_MAP = {"r": "Rock", "p": "Paper", "s": "Scissors"}


def main():
    if not DATA_PATH.exists():
        print(f"Missing dataset: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    y = df["label"].map(LABEL_MAP)
    X = df.drop(columns=["label"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nReport:")
    print(classification_report(y_test, preds))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()