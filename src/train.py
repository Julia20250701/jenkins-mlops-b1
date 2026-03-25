from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def main():
    data_path = Path("data/iris.csv")
    model_dir = Path("models")
    result_dir = Path("results")

    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, model_dir / "model.joblib")

    X_test.to_csv(result_dir / "X_test.csv", index=False)
    y_test.to_csv(result_dir / "y_test.csv", index=False)

    print("Training finished.")
    print("Saved model to models/model.joblib")
    print("Saved test split to results/X_test.csv and results/y_test.csv")


if __name__ == "__main__":
    main()
