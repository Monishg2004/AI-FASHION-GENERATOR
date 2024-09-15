import base64
from flask import Flask, render_template, jsonify
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

pro = Flask(__name__)

new_model = tf.keras.models.load_model('genetor.h5')

@pro.route('/')
def index():
    return render_template('pro2.html')

@pro.route('/generate_images', methods=['GET'])
def generate_images():
    imgs = new_model.predict(tf.random.normal((100, 128)))
    img_data = []

    for img_array in imgs:
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array.squeeze()) 
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_data.append(base64.b64encode(img_bytes.getvalue()).decode('utf-8'))

    return jsonify(images=img_data)

if __name__ == '__main__':
    pro.run(debug=True)
