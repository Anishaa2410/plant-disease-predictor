import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import img_to_array
import cv2
import pickle
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)


model =load_model('model/my_model.h5', compile=False)
print('Model loaded. Check !')

default_image_size = tuple((256, 256))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def model_predict(img_path, model):
    image_array=convert_image_to_array(img_path)
    np_image=np.array(image_array, dtype=np.float16)/225.0
    np_image=np.expand_dims(np_image,0)
    plt.imshow(plt.imread(img_path))
    result=np.argmax(model.predict(np_image), axis=-1)
    return result

  


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        filename=r"C:\Users\ANISHA\Desktop\Project\main\model\label_transform.pkl"
        image_labels=pickle.load(open(filename,'rb'))
        final= image_labels.classes_[preds][0]
        print(final)
        return final
    return None


if __name__ == '__main__':
    # app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
