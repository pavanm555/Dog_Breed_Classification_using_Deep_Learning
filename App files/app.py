import numpy as np
import os
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)
model = load_model("DogBreeds.h5")
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(299,299))
        breeds = ['Boxer',
 'Dachshund',
 'Dalmatian',
 'Indian Spitz',
 'Shis Tzu',
 'Siberian husky',
 'afghan_hound',
 'african_hunting_dog',
 'airedale',
 'basenji',
 'basset',
 'beagle',
 'bedlington_terrier',
 'bernese_mountain_dog',
 'black-and-tan_coonhound',
 'blenheim_spaniel',
 'bloodhound',
 'bluetick',
 'border_collie',
 'border_terrier',
 'borzoi',
 'boston_bull',
 'bouvier_des_flandres',
 'brabancon_griffon',
 'bull_mastiff',
 'cairn',
 'cardigan',
 'chesapeake_bay_retriever',
 'chow',
 'clumber',
 'cocker_spaniel',
 'collie',
 'curly-coated_retriever',
 'dhole',
 'dingo',
 'doberman',
 'english_foxhound',
 'english_setter',
 'entlebucher',
 'flat-coated_retriever',
 'german_shepherd',
 'german_short-haired_pointer',
 'golden_retriever',
 'gordon_setter',
 'great_dane',
 'great_pyrenees',
 'groenendael',
 'ibizan_hound',
 'irish_setter',
 'irish_terrier',
 'irish_water_spaniel',
 'irish_wolfhound',
 'japanese_spaniel',
 'keeshond',
 'kerry_blue_terrier',
 'komondor',
 'kuvasz',
 'labrador_retriever',
 'leonberg',
 'lhasa',
 'malamute',
 'malinois',
 'maltese_dog',
 'mexican_hairless',
 'miniature_pinscher',
 'miniature_schnauzer',
 'newfoundland',
 'norfolk_terrier',
 'norwegian_elkhound',
 'norwich_terrier',
 'old_english_sheepdog',
 'otterhound',
 'papillon',
 'pekinese',
 'pembroke',
 'pomeranian',
 'pug',
 'redbone',
 'rhodesian_ridgeback',
 'rottweiler',
 'saint_bernard',
 'saluki',
 'samoyed',
 'schipperke',
 'scotch_terrier',
 'scottish_deerhound',
 'sealyham_terrier',
 'shetland_sheepdog',
 'standard_poodle',
 'standard_schnauzer',
 'sussex_spaniel',
 'tibetan_mastiff',
 'tibetan_terrier',
 'toy_terrier',
 'vizsla',
 'weimaraner',
 'whippet',
 'wire-haired_fox_terrier',
 'yorkshire_terrier']
        img = image.img_to_array(np.squeeze(img))
        img = np.expand_dims(img, axis=0) 
        img /= 255.0 
        pred = np.argmax(model.predict(img))
        output = breeds[pred]
        output = re.sub('[^a-zA-Z]',' ',output)
        output = output.title()
        return render_template("output.html", prediction=output)

if __name__=='__main__':
    app.run()