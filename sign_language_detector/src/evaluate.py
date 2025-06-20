from sign_language_detector.src.improved_preprocess import get_test_data
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np

X_test, y_test = get_test_data("data/test.csv")

model = load_model("saved_models/model.h5")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))
