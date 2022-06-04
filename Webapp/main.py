from flask import Flask, render_template, request,  redirect, flash
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys
from keras.models import model_from_json
import skimage.transform
import json
import scipy
UPLOAD_FOLDER = './assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='/assets',
            static_folder='./assets')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@ app.route('/')
def root():
    return render_template('index.html')


@ app.route('/index.html')
def index():
    return render_template('index.html')


@ app.route('/about.html')
def about():
    return render_template('about.html')


@ app.route('/movie.html')
def upload():
    return render_template('movie.html')


@ app.route("/show", methods=['POST', 'GET'])
def uploaded_chest():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'movie.jpg'))

    model = model_from_json(open("gpre.json", "r").read())
    model.load_weights("gpre.h5")

    image_name = "the_conjuring.jpeg"
    image_path = "test-images/"+image_name
    img = cv2.imread(image_path)
    #image = img.copy()

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

    ljson = open("label.json", "r")
    labels = json.load(ljson)
    for idx in ids:
        print(labels['id2genre'][idx], predictions[0][idx])
    ljson.close()
    result1 = labels['id2genre'][idx], predictions[0][idx]
    return render_template('show.html', prob=result1)


def preprocessImg(img, size):
    img = skimage.transform.resize(img, size)
    img = img.astype(np.float32)
    #img = (img/127.5)-1
    # print(img)
    return img


if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)
