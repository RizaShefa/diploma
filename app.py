import os
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import cv2 # type: ignore
from keras.models import load_model # type: ignore
from flask import Flask, request, render_template # type: ignore
from werkzeug.utils import secure_filename # type: ignore

app = Flask(__name__)

model = load_model('BreastCancer10Epochs.h5')



def get_className(classNo):
    if classNo == 0:
        return "No  Breast Cancer"
    elif classNo == 1:
        return "Yes Breast Cancer"


def getResult(img):
    image_path = os.path.join(os.path.dirname(__file__), img)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    return result


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/home')
def home_redirect():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/appointment')
def appointment():
    return render_template('appointment.html')





@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'pred', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(np.argmax(value))
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
