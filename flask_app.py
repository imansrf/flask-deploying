from __future__ import division, print_function
# coding=utf-8
import sys
import glob
import re
import numpy as np
import os
from flask import Flask, render_template, Response, url_for, redirect
from flask import request

# Keras
from keras.preprocessing import image
from keras.models import load_model
import h5py


app =  Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/CP.h5')

model = load_model(MODEL_PATH)
print('Model loaded')

def model_predict(img_path, model):
    img = image.load_img(img_path,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    result = model.predict(img)
    for i in result:
        index = np.argmax(i)
        if index == 0:
            text = "NEGATIVE" 
        elif index == 1:
            text = "POSITIVE" 
        return text


@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 OCCURED. Page not found. Please go the home page and try again"
    return render_template("error.html", message=message)

@app.errorhandler(405)
def error405(error):
    message = "Error 405, Method not found"
    return render_template("error.html",message=message)
  
@app.errorhandler(500)
def error500(error):
    message = "INTERNAL ERROR 500, Error occurs in the program"
    return render_template("error.html",message=message)


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename
        print('The filename that has been uploaded =',filename)
        # know the extension of filename
        # all only .jpg, .png, .jpeg
        ext = filename.split('.')[-1]
        print('The extension of the filename =',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            # saving the image
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved succesfully')
            # Make prediction
            preds = model_predict(path_save,model)
            print(preds)
            return render_template('upload.html',prediction=preds,extension=False,fileupload=True,data=preds,image_filename=filename)
           
        else:
            print('Use only the extension with .jpg, .png, .jpeg ')
            return render_template('upload.html', extension=True,fileupload=False)

    else: 
        return render_template('upload.html',fileupload=False,extension=False)

@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)