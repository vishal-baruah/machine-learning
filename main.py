"""Recognize handwritten digits with a multilayer perceptron."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def main():
    digits = load_digits()
    images = digits.images
    features = digits.data / 16.0
    labels = digits.target

    x_train, x_test, y_train, y_test, _, test_images = train_test_split(
        features,
        labels,
        images,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
    )
    axes[0].set_title("Test-set confusion matrix")
    axes[1].imshow(test_images[0], cmap="gray_r")
    axes[1].set_title(f"Predicted: {predictions[0]} | Actual: {y_test[0]}")
    axes[1].axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()