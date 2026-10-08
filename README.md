# Handwritten Digit Recognition

Train a feed-forward neural network to recognize handwritten digits using
scikit-learn's built-in digits dataset. Each example is an 8 x 8 grayscale
image, and the network predicts a digit from 0 to 9.

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

The program prints test accuracy and a classification report, then displays a
confusion matrix and an example prediction. The data is included with
scikit-learn, so no separate dataset download is needed.
