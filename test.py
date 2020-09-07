from keras.models import load_model
import cv2
import numpy as np
import sys

def mapper(index):
    return CLASS_REV_MAP[index]

CLASS_REV_MAP = {
    0: "rock",
    1: "paper",
    2: "scissor",
    3: "none"
}

filepath = sys.argv[1]
model = load_model("rock_paper_scissor.h5")

img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227,227))

pred = model.predict_classes(np.array([img]))
pred = mapper(pred[0])
print(pred)