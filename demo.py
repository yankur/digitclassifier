import numpy as np
from digitclassifier import DigitClassifier

image = np.random.rand(28, 28)  # Example image

# For CNN
cnn_classifier = DigitClassifier(algorithm="cnn")
print(cnn_classifier.predict(image))

# For Random Forest
rf_classifier = DigitClassifier(algorithm="rf")
print(rf_classifier.predict(image))

# For Random
rand_classifier = DigitClassifier(algorithm="rand")
print(rand_classifier.predict(image))