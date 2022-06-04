import numpy as np
import cv2
import sys
from keras.models import model_from_json
import skimage.transform
import json
import scipy


# load saved model
model = model_from_json(open("gpre.json", "r").read())
model.load_weights("gpre.h5")


def preprocessImg(img, size):
    img = skimage.transform.resize(img, size)
    img = img.astype(np.float32)
    #img = (img/127.5)-1
    # print(img)
    return img


if __name__ == "__main__":
    image_name = "sr.jpg"
    image_path = "test-images/"+image_name
    img = cv2.imread(image_path)
    image = img.copy()
    img_size = (150, 100, 3)
    img = preprocessImg(img, img_size)
    img = np.array(img, 'float32')
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)

    predictions = np.array(predictions)
    # print(predictions)
    ids = predictions[0].argsort()[::-1]
    # print(ids)
    ids = ids[:7]
    # print(ids)

    ljson = open("label.json", "r")
    labels = json.load(ljson)
    for idx in ids:
        print(labels['id2genre'][idx], predictions[0][idx])
    ljson.close()
    while(1):
        cv2.imshow("Movie: ", image)
        if cv2.waitKey(0) == ord('q'):
            break
