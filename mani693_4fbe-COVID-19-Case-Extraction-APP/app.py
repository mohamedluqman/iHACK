# Imported Modules
from flask import Flask, render_template, request, jsonify

import requests, json, os
from newspaper import Article
from text2digits import text2digits

import spacy, random
from spacy.tokens import Span

app = Flask(__name__)    
SITE_ROOT = os.path.realpath(os.path.dirname(__file__))

@app.route('/')   
def main():
    return render_template('index.html')

# Api Call to get url
@app.route('/api/getdata',methods=["GET"])   
def getData():
    url = request.args.get('url')
    # From Newspaper Framework getting required data
    content = Article(url) 
    content.download()
    content.parse()
    title = content.title
    rawText = content.text
    # Unformatted Data to show to user
    textDisplay = rawText.split("\n\n")
    textDisplay = ''.join(textDisplay)
    # Converting numbered text to digits
    t2d = text2digits.Text2Digits()
    numText=t2d.convert(rawText)
    text = numText.split("\n\n")
    text = ''.join(text)
    # Implemented API data limit restriction 
    if len(text) < 5000:
        text = text
    else:
        text = text[:5000]
    jsonData = {"text" : text}
    configDataResource = os.path.join(SITE_ROOT, "data", "configdata.json")
    configData = json.load(open(configDataResource))

    # NER API call request
    headers = {'x-api-key': configData["X_API_KEY"], 'Content-type': 'application/json'}
    ner_response = requests.post(configData["NAMED_ENTITY_RECOGNITION_ENDPOINT"], headers=headers, data=json.dumps(jsonData))
    # print(ner_response.text)
    # Deserializing the response
    places = lambda:None
    places.__dict__ = json.loads(ner_response.text)
    print(places.LOC)

    
    json_url = os.path.join(SITE_ROOT, "data", "sg-citi.json")
    data = json.load(open(json_url))

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    LOC = []
    CASE = []
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
        if ent.label_ == "CARDINAL":
            CASE.append(ent.text)
        if ent.label_ == "GPE":
            LOC.append(ent.text)

    count  = []
    for i in CASE:
        if i.isdigit():
            if i not in count:
                count.append(i)
    print("COUNT: ",count)
    if not len(count):
        count = list(i for i in range(80, 500, 7))
    returnJson = {"text": textDisplay, "location": [], "category" : ner_response.text}
    for i in places.LOC:
        for citi in data:
            if i in citi["name"] and citi["name"] not in returnJson["location"]:
                returnJson["location"].append({"name" : citi["name"], "lat": "no1", "lon": "no2", "count" : count[random.randrange(0,len(count))]})
                break
    print(returnJson)
    return jsonify(returnJson)

# Api Call to train Model
@app.route('/api/trainmodule',methods=["POST"])   
def trainModule():
    # Model Training
    rawData = request.args.get('rawData')
    train_data = request.args.get('trainData')
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(rawData)
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
    for i in range(50):
        random.shuffle(train_data)
        for text, annotations in train_data:
            nlp.update([text], [annotations], sgd=optimizer)
    nlp.to_disk("/model")
    return "{'status' : 'Success', 'message': 'model train'}"

if __name__ == '__main__':  
    app.run(debug=True) 




