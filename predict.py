import pickle
import sys
import warnings

import numpy as np
warnings.filterwarnings("ignore", message="X does not have valid feature names")

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "alone"]

EXAMPLES = [
    ("Мужчина, 22 года, 3-й класс, едет один", [3, 0, 22, 0, 0, 7.25, 1]),
    ("Женщина, 38 лет, 1-й класс, с супругом", [1, 1, 38, 1, 0, 71.28, 0]),
]


def main():
    with open("/app/model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Модель загружена:", type(model).__name__)

    args = sys.argv[1:]
    if args:
        if len(args) != len(FEATURES):
            sys.exit(f"Нужно {len(FEATURES)} значений: {' '.join(FEATURES)}")
        passengers = [("Пассажир из аргументов", [float(a) for a in args])]
    else:
        passengers = EXAMPLES

    X = np.array([values for _, values in passengers], dtype=float)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    for (title, values), pred, proba in zip(passengers, predictions, probabilities):
        print(f"\n{title}")
        print("  признаки:", dict(zip(FEATURES, values)))
        print(f"  предсказание: {'выжил' if pred == 1 else 'не выжил'}"
              f" (вероятность выжить {proba:.2f})")


if __name__ == "__main__":
    main()
