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
import random
cors = CORS(app)

client = MongoClient("mongodb+srv://musa:1221@cluster0.ps9aijg.mongodb.net/test")
db = client.users
grid_fs = gridfs.GridFS(db)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Loading TROCR')

import warnings
from contextlib import contextmanager
from transformers import MBartTokenizer, ViTImageProcessor, XLMRobertaTokenizer
from transformers import ProcessorMixin


class CustomOCRProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

from transformers import TrOCRProcessor

image_processor = ViTImageProcessor.from_pretrained(
    'microsoft/swin-base-patch4-window7-224-in22k'
)
tokenizer = MBartTokenizer.from_pretrained(
    'facebook/mbart-large-50'
)
processortext = CustomOCRProcessor(image_processor,tokenizer)

image_processor = ViTImageProcessor.from_pretrained(
    'microsoft/swin-base-patch4-window12-384-in22k'
)
tokenizer = MBartTokenizer.from_pretrained(
    'facebook/mbart-large-50'
)
processortext2 = CustomOCRProcessor(image_processor,tokenizer)

model = VisionEncoderDecoderModel.from_pretrained(r"C:\Users\MusaDAC\Downloads\MedicalViLanOCR").to(device)
model2 = VisionEncoderDecoderModel.from_pretrained("musadac/vilanocr-single-urdu",use_auth_token=True).to(device)

#microsoft/trocr-large-handwritten
#./trocr-trained-best
print('Loaded TROCR ✅')

################################### Layout LM requirements #################################################################
from transformers import LayoutLMv3ForTokenClassification
from transformers import AutoProcessor

#### Some necessary variables  ################

labels = ['B-MEDICINE DOSE','I-MEDICINE DOSE','B-DIAGNOSIS','I-DIAGNOSIS','B-HISTORY','I-HISTORY',\
            'B-BP','B-MEDICINE TYPE','B-MEDICINE NAME','B-MEDICINE POWER','B-NAME','B-GENDER','B-DATE','B-AGE',\
                'B-TEMP','B-WEIGHT']


label2color = {lb:f"rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},0.3)"  for lb in labels}
label2idx ={lb:idx  for idx,lb in enumerate(labels)}
idx2lb ={idx:lb  for idx,lb in enumerate(labels)}
num_labels = len(labels)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "model_layout-1"    ### LayoutLM path
max_length = 128

#################################################


## Utility function
def normalize_the_bbox(bbox,img_width,img_height):
    """_summary_

    Args:
        bbox (_array_): _the array contains coordinates like (xmin,ymin,xmax,ymax)_

    Returns:
        _array_: _returns normalized arrays_
        
    """
    if bbox[2]>=img_width:
        bbox[2] = img_width-1
    if bbox[3] >=img_height:
        bbox[3] = img_width-1

    if bbox[0] <=0:
        bbox[0] = 1
    
    if bbox[1] <=0:
        bbox[1] = 1
    
    bbox = [
        (bbox[0]*1000)/img_width,
        (bbox[1]*1000)/img_height,
        (bbox[2]*1000)/img_width,
        (bbox[3]*1000)/img_height
    ]
    
    for i in range(len(bbox)):
        if bbox[i]>=1000:
            bbox[i]=999
        if bbox[i] <=0:
            bbox[i] = 1
    
    return bbox



### Initializing the model ####
processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False)

layoutmodel = LayoutLMv3ForTokenClassification.from_pretrained(model_path,
                                                          num_labels=num_labels)


layoutmodel.to(device)
print("Loading the layout LM model")

##############################################################################################################################
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
        pixel_values = processortext2(i.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values.to(device))
        generated_text.append(processortext2.batch_decode(generated_ids, skip_special_tokens=True)[0])
        # if(temp['localization'][count]['label_name'] == 'English'):
        #     generated_ids = model.generate(pixel_values.to(device))
        # else:
        #     generated_ids = modelUrdu.generate(pixel_values.to(device))
        # generated_text.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
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


### Layout LM API
@app.route("/extract_entities",methods = ['POST'])
def prediction():
    
    """_Input_
    request ={"entity_recogn":
                    {
                        "image" :image_file
                        "bbox":array of bounding boxxes with (xmin,ymin,xmax,ymax) format
                        "text": array of text corresponding to the bounding box
                    }
    }

    Returns:
        _type_: _description_
    """
    input_obj = request.json['entity_recogn']  ### Fetching the object
    
    ## Reading image as base64 bytes
    im_b64 = input_obj["image"]
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    image_file = Image.open(io.BytesIO(img_bytes))
    
    
    img_width,img_height = image_file.width,image_file.height
    
    text =  input_obj['text']
    bboxes =  input_obj['bbox']
    
    normalize_bbox = [normalize_the_bbox([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]],img_width=img_width,img_height=img_height) for bbox in bboxes]
    
    tmp_labels = [1 for i in range(len(text))]
    
    encoding = processor(image_file, text, boxes=normalize_bbox, word_labels=tmp_labels, return_tensors="pt",\
                                max_length =max_length, padding ="max_length",truncation=True)
    
    lb = encoding.pop("labels")
    
    
    for k,v in encoding.items():
        encoding[k] = v.to(device)
        
        
    encoding['input_ids'] = encoding['input_ids'].long() 
    encoding['bbox'] = encoding['bbox'].long()
    
    output = layoutmodel(**encoding) 
    
    
    pred = output.logits.argmax(-1).detach().cpu()[0]
    
    
    predictions =[ idx2lb[k.item()] for k,v in zip(pred,lb[0]) if v != -100 ]
    
    
    return {"predictions":predictions,"bbox":bboxes,"text":text}


@app.route("/extract_text_entities",methods = ['POST'])
def endtoend():
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
        pixel_values = processortext2(i.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values.to(device))
        generated_text.append(processortext2.batch_decode(generated_ids, skip_special_tokens=True)[0])



    img_width,img_height = image.width,image.height
    
    text =  generated_text
    bboxes = []
    fullloc = []
    for i in temp['localization']:
        bboxes.append(i['bbox'])
        fullloc.append(i)
    
    normalize_bbox = [normalize_the_bbox([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]],img_width=img_width,img_height=img_height) for bbox in bboxes]
    
    tmp_labels = [1 for i in range(len(text))]
    print(len(bboxes))
    encoding = processor(image.convert("RGB"), text, boxes=normalize_bbox, word_labels=tmp_labels, return_tensors="pt", max_length =max_length, padding ="max_length",truncation=True)
    
    lb = encoding.pop("labels")
    
    
    for k,v in encoding.items():
        encoding[k] = v.to(device)
        
        
    encoding['input_ids'] = encoding['input_ids'].long() 
    encoding['bbox'] = encoding['bbox'].long()
    
    output = layoutmodel(**encoding) 
    
    
    pred = output.logits.argmax(-1).detach().cpu()[0]
    
    
    predictions =[ idx2lb[k.item()] for k,v in zip(pred,lb[0]) if v != -100 ]
    
    colors = [label2color[i]  for i in predictions]
    
    products.update_one({"_id" : temp['_id']}, {"$set" : {"generated_text" :generated_text}})
    products.update_one({"_id" : temp['_id']}, {"$set" : {"predictions" :predictions}})
    products.update_one({"_id" : temp['_id']}, {"$set" : {"colors" :colors}})
    
    return {"localization":fullloc, "predictions":predictions,"bbox":bboxes,"text":text,"colors":colors}

@app.route("/hit",methods = ['POST'])
def hit():
    return {}

import os
@app.route("/urdu",methods = ['POST'])
def urdu():
    file = request.files['image']
    if file:
        filename = file.filename
        file.save(filename) 
        image_path = os.path.join(filename)
        i = Image.open(image_path)
        pixel_values = processortext2(i.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = model2.generate(pixel_values.to(device))
        os.remove(filename)
        return {'text':processortext2.batch_decode(generated_ids, skip_special_tokens=True)[0]}

if __name__ == "__main__":
    app.run(host='0.0.0.0',port = 5000)
