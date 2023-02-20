from yolov5 import run
from yolov5 import app,secure_filename,request,allowed_file
from PIL import Image
import pymongo
from pymongo import MongoClient
import gridfs
from flask import Flask, request, jsonify
from datetime import datetime


import torch
import os

imgsz = [416,416]

conf_thres =0.4

client = MongoClient("mongodb+srv://musa:1221@cluster0.ps9aijg.mongodb.net/test")
db = client.users
grid_fs = gridfs.GridFS(db)
products = db.docigize



def saveImage(filename,cat,local):
    with open(filename, 'rb') as f:
        image = f.read()
    name = filename
    name = name.split('/')[-1]
    id = grid_fs.put(image, filename = name)
    query = {
        'id':id,
        'name':name ,
        'desc':cat,
        'localization':local,
        'time':datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    }
    status = products.insert_one(query)
    if status:
        return id
    return jsonify({'result': 'Error occurred during uploading'}),500

@app.route("/detect_bounding_box",methods = ['POST'])
def fetch_bounding_box():
    """
    Input: {"image":imagefile}
    
    returns:
        {
            "localization":list[{"confidence":float,
                                "label_name":string,
                                "bbox":[xmin,ymin,xmax,ymax]}]
        }
    """    


    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))      
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  

        result = run(source=image_path,imgsz=imgsz,conf_thres=conf_thres)
        id = saveImage(image_path,request.form['email'],result)
        os.remove(image_path)
        return {"_id":str(id),"localization":result}
    

    return {"localization":"Incorrect File Format"}    


