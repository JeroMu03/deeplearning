import pickle
import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset

# Cargar dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Cargar modelo ya entrenado
with open("modelo_entrenado.pkl", "rb") as f:
    d = pickle.load(f)

index = 10,

plt.imshow(test_set_x_orig[index])
print(
    "y = " + str(test_set_y[0, index]) +
    ", you predicted that it is a \"" +
    classes[int(d['Y_prediction_test'][0, index])].decode("utf-8") +
    "\" picture."
)

plt.show()