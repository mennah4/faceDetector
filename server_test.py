		
import cv2 
import numpy as np
from flask import Flask, request, make_response,jsonify
import numpy as np
import json
import urllib.request
from urllib.request import Request, urlopen
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import logging
import imutils
import pickle
import os
import argparse
import datetime


model = None
app = Flask(__name__,static_url_path='')

def send_mail(time,address,name):
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "mennahjafar@hotmail.com"
    toaddr = ["mennahjafar@hotmail.com"]

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = ""

    # storing the subject
    msg['Subject'] = "Test"

    # string to store the body of the mail
    body = "{} was detected in Loc {} at Time {} ".format(name,address,time)

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # open the file to be sent
    #filename = img
    #attachment = open(filename, "rb")

    # instance of MIMEBase and named as p
    #p = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    #p.set_payload((attachment).read())

    # encode into base64
    #encoders.encode_base64(p)

    #p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    #msg.attach(p)

    # creates SMTP session
    s = smtplib.SMTP('smtp-mail.outlook.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "skyfallM4")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()






ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

def preprocess_img(img,target_size=(299,299)):
	image = imutils.resize(img, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
	return image,h,w,imageBlob


def load_im_from_system(url):
    image_url = url.split(',')[1]
    image_url = image_url.replace(" ", "+")
    image_array = base64.b64decode(image_url)
    #image_array = np.asarray(bytearray(image_array), dtype=np.uint8)
    image_array = np.fromstring(image_array, np.uint8)
    image_array = cv2.imdecode(image_array, -1)
    return image_array    

def predict(img):
    image,h,w,imageBlob=preprocess_img(img)
    print (img.shape)
    global model
    if model is None:
        print("[INFO] loading face detector...")
        protoPath = os.path.join("face_detection_model", "deploy.prototxt")
        modelPath = os.path.join("face_detection_model","res10_300x300_ssd_iter_140000.caffemodel")
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        print("[INFO] loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

        recog = open("output/recognizer.pickle", "rb")
        recognizer = pickle.load(recog)
        le = pickle.loads(open("output/le.pickle", "rb").read())
    detector.setInput(imageBlob)
    detections = detector.forward()
    names = []
    probabilities = []
    ts1 = []
    address1 = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),(0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            print("Name:", name)
            print("Proba", proba)
            names.append(name)
            current_time = datetime.datetime.now()
            ts = current_time.strftime("%A %d %B %Y %I:%M:%S%p")
            address = "Turkey"
            ts1.append(ts)
            address1.append(address)

            probabilities.append(proba)
            #text = "{}: {:.2f}%".format(name, proba * 100)
            #y = startY - 10 if startY - 10 > 10 else startY + 10
            #cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
            #cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return names, probabilities,address1,ts1



@app.route('/classify_system', methods=['GET'])
def classify_system():
    image_url = request.args.get('imageurl')
    image_array = load_im_from_system(image_url)
    names, probabilities,address1,ts1 = predict(image_array)
    
    result = []
    for r in range(len(names)):
        result.append({"class_name":names[r],"score":float(probabilities[r]),"address":address1[r],"ts":ts1[r]})
        send_mail(ts1[r], address1[r], names[r])
    return jsonify({'results':result})


@app.route('/classify-system', methods=['GET'])
def do_system():
    return app.send_static_file('system.html')


@app.route('/', methods=['GET'])
def root():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(port=8000, debug=True)

