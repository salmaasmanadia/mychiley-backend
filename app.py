from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import json
from bson import json_util

from pymongo import MongoClient

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
print("successfully running")

from dotenv import load_dotenv
load_dotenv()
MONGO_URI = os.environ.get('MONGO_URI')
cluster = MongoClient(MONGO_URI)
db      = cluster['my-chiley']
col     = db['histories']
import cloudinary
import cloudinary.uploader
import cloudinary.api
config = cloudinary.config(secure=True)
print("****1. Set up and configure the SDK:****\nCredentials: ", config.cloud_name, config.api_key, "\n")

import logging

# cors library
from flask_cors import CORS
# Define a flask app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "http://localhost:3000",
        "methods": ["GET", "POST"]
    }
})

@app.get("/")
async  def home():
    return {'message': 'Welcome to our API'}

# Path ke model yang telah dilatih
MODEL_PATH = 'models/chiley-model.pth'

# Load model yang telah dilatih
model = models.resnet50(pretrained=False)  # Sesuaikan dengan jumlah kelas Anda
num_classes = 2 # Ubah sesuai dengan jumlah kelas dari model Anda
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Sesuaikan layer output

# Load the trained model state dict
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# Update state dict to match the current model definition
model_state_dict = model.state_dict()
for key, value in state_dict.items():
    if key in model_state_dict and value.size() == model_state_dict[key].size():
        model_state_dict[key] = value

model.load_state_dict(model_state_dict)
model.eval()

print('Model loaded. Check http://127.0.0.1:5000/')

# Definisikan nama-nama kelas dan reverse mapping
Name = ['Healthy', 'notHealthy']
N = [i for i in range(len(Name))]
reverse_mapping = dict(zip(N, Name))

def model_predict(img_path, model):
    # Preprocess gambar
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path)
    img = image_transform(img)
    img = img.unsqueeze(0)  # Tambahkan dimensi batch

    # Lakukan prediksi
    with torch.no_grad():
        preds = model(img)
        preds = torch.softmax(preds, dim=1)  # Menggunakan softmax untuk probabilitas kelas

    print(preds)
    return preds

def mapper(value):
    # Konversi tensor ke integer sebelum melakukan mapping
    value = value.item()
    return reverse_mapping[value]


@app.route('/prediction', methods=['GET', 'POST'])
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
        
        response = cloudinary.uploader.upload(file_path)
        logging.info("response", response)
        
        value = np.argmax(preds)
        move_name = mapper(value)
        result = {"prediction": move_name}
        result["image_url"] = str(response["secure_url"])
        result["email"] = request.form.get("email")
        result["date"] = request.form.get("date")
        result_json = json.dumps(result)
        print(value)
        print("last step")
        
        # Save to histories
        client = MongoClient(MONGO_URI)

        db = client["my-chiley"]
        collection = db["histories"]
        
        data = {
        "url"          : result["image_url"],
        "email"        : result["email"],
        "date"         : result["date"],
        "prediction"   : move_name
        }
        print(data)
        
        collection.insert_one(data)

        client.close()

        return result_json
    elif request.method == 'GET':
        # Get the file from post request
        email = request.args.get('email')
    
        client = MongoClient(MONGO_URI)

        db = client["my-chiley"]
        collection = db["histories"]

        # Create a query object to match the UID
        query = {'email': email}
        print(email)
        # Use the find method to retrieve all matching documents
        cursor = collection.find(query)

        # Convert the documents to a list
        histories = list(cursor)
        new_histories = []

        for history in histories:
            # Convert the ObjectId to a string
            history['_id'] = str(history['_id'])
            new_histories.append(history)

        # Close the MongoDB connection
        client.close()

        result = {
            'histories': new_histories
        }

        result_json = json.dumps(result, default=json_util.default)

        return result_json
    return None



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)