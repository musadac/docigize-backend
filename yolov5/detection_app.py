from yolov5 import run
from yolov5 import app,secure_filename,request,allowed_file
from PIL import Image
import pymongo
from pymongo import MongoClient
from flask import Flask, request, jsonify
from datetime import datetime
from bson import ObjectId
import gridfs
import base64
import io
import torch
import os

imgsz = [416,416]

conf_thres =0.4

client = MongoClient("mongodb+srv://<>:<>@<>")
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
    print(status)
    if status:
        return status.inserted_id
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
        
        try:
            result = sorted(sorted(result,key= lambda x: x['bbox'][0]),key = lambda x: x['bbox'][1]) ## Sorting the yolo results
        except:
            print(f"Error in sorting")
            print(f"Using unsorted values")
        print(request.files['email'])
        id = saveImage(image_path,request.form['email'],result)
        os.remove(image_path)

        products = db.docigize
        image_data = ""
        for i in products.find({'desc':request.files['email'], '_id':ObjectId(id)}):
            image = grid_fs.get(i['id'])
            imageStream = io.BytesIO(image.read())
            image_data = base64.b64encode(imageStream.getvalue()).decode('utf-8')
            break

        return {"_id":str(id),"localization":result, 'image_data':image_data}
    

    return {"localization":"Incorrect File Format"}    


