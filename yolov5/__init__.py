from flask import Flask,jsonify, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

import os



app = Flask(__name__)
CORS(app, support_credentials=True)


app.secret_key = "secret key" 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])



########################## Function to check if file allowed ###################################
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from yolov5.detect import *
from yolov5 import detection_app