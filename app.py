from yolov5 import app
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from datetime import datetime
import pymongo
from pymongo import MongoClient
import gridfs
import base64
import io
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from bson import ObjectId
from IPython.display import display
import torch
cors = CORS(app)

client = MongoClient("mongodb+srv://musa:1221@cluster0.ps9aijg.mongodb.net/test")
db = client.users
grid_fs = gridfs.GridFS(db)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Loading TROCR')




processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
processorUrdu = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
processorComb = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)
modelUrdu = VisionEncoderDecoderModel.from_pretrained("/Users/musadac/Downloads/NewWeight5").to(device)
modelCombined = VisionEncoderDecoderModel.from_pretrained("/Users/musadac/Downloads/EngUrdu5").to(device)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 10
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 3

modelUrdu.config.decoder_start_token_id = processorUrdu.tokenizer.cls_token_id
modelUrdu.config.pad_token_id = processorUrdu.tokenizer.pad_token_id
modelUrdu.config.vocab_size = modelUrdu.config.decoder.vocab_size
modelUrdu.config.eos_token_id = processorUrdu.tokenizer.sep_token_id
modelUrdu.config.max_length = 10
modelUrdu.config.early_stopping = True
modelUrdu.config.no_repeat_ngram_size = 3
modelUrdu.config.length_penalty = 2.0
modelUrdu.config.num_beams = 3

modelCombined.config.decoder_start_token_id = processorComb.tokenizer.cls_token_id
modelCombined.config.pad_token_id = processorComb.tokenizer.pad_token_id
modelCombined.config.vocab_size = modelCombined.config.decoder.vocab_size
modelCombined.config.eos_token_id = processorComb.tokenizer.sep_token_id
modelCombined.config.max_length = 10
modelCombined.config.early_stopping = True
modelCombined.config.no_repeat_ngram_size = 3
modelCombined.config.length_penalty = 2.0
modelCombined.config.num_beams = 3
#microsoft/trocr-large-handwritten
#./trocr-trained-best
print('Loaded TROCR ✅')

@app.route('/login',methods = ['POST', 'GET'])
def login():
    products = db.creds
    data = request.json
    products.insert_one({
    'name': 'Musa Cheema',
    'email': 'mcheema2010@gmail.com',
    'password': '1234',
    })
    return jsonify(data)




def cropimages(img, bbox):
    imgs = []
    for i in range(len(bbox)):
        box = (
            float(bbox[i]['bbox'][0]),
            float(bbox[i]['bbox'][1]),
            float(bbox[i]['bbox'][2]),
            float(bbox[i]['bbox'][3]))
        img2 = img.crop(box)
        imgs.append(img2)
    return imgs

@app.route('/get_text',methods = ['POST'])    
def get_text():
    products = db.docigize
    data = request.json
    data = products.find({'_id':ObjectId(data['id'])})
    temp = ""
    for i in data:
        temp = i
    image = grid_fs.get(temp['id'])
    image = Image.open(io.BytesIO(image.read()))
    allimg = cropimages(image, temp['localization'])
    generated_text = []
    count = 0
    for i in allimg:
        
        # if(temp['localization'][count]['label_name'] == 'English'):
        #     generated_ids = model.generate(pixel_values.to(device))
        # else:
        #     generated_ids = modelUrdu.generate(pixel_values.to(device))
        # generated_text.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
        generatedtext = ""
        pixel_values = processor(i.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values.to(device))
        generatedtext += processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generatedtext += ", "

        pixel_values = processorComb(i.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = modelCombined.generate(pixel_values.to(device))
        generatedtext += processorComb.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generatedtext += ", "

        pixel_values = processorUrdu(i.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = modelUrdu.generate(pixel_values.to(device))
        generatedtext += processorUrdu.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text.append(generatedtext)
    products.update_one({"_id" : temp['_id']}, {"$set" : {"generated_text" :generated_text}})
    return {'generated_text':generated_text}


@app.route("/all_docs",methods = ['POST'])
def get_image():
#    data = request.json
    products = db.docigize
    alldoc = []
    for i in products.find({'desc':'mcheema2010@gmail.com'}):
        alldoc.append(i)
    for i in alldoc:
        image = grid_fs.get(i['id'])
        i['_id'] = str(i['_id'])
        i['id'] = ""
        imageStream = io.BytesIO(image.read())
        image_data = base64.b64encode(imageStream.getvalue()).decode('utf-8')
        i['image'] = image_data
    return {'alldoc': alldoc}





if __name__ == "__main__":
    app.run(port = 5000,debug = True,)
