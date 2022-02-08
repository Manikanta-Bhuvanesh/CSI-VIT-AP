#!curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok   
#!ngrok authtoken XXXXXXXXXXXXXXXXX
#!pip install flask_ngrok
import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask_ngrok import run_with_ngrok


app = Flask(__name__)
run_with_ngrok(app) 


from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
from keras.preprocessing import image

print("Loading model") 
global sess
sess = tf.compat.v1.Session()
set_session(sess)
global model 
model = load_model('my_model.h5') 


@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>') 
def prediction(filename):
      set_session(sess)
      img= image.load_img(os.path.join('uploads', filename), target_size=(224, 224))
      img=image.img_to_array(img)
      img=np.expand_dims(img,axis=0)
      probabilities = model.predict(img)
      print(probabilities)
      number_to_class = ['Car','Plane']
      index = np.argmax(probabilities,axis=1)
      predictions = {
        "class":number_to_class[index[0]],
      }
      return render_template('predict.html', predictions=predictions)


app.run()
