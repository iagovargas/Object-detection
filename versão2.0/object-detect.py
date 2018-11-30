#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:36:42 2018

@author: iago
"""

# import the necessary packages
from twilio.rest import Client
from imutils.video import VideoStream
from imutils.video import FPS
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import numpy as np
import imutils
import time
import cv2
import smtplib


def sendemail():
    msg = MIMEMultipart()
    message = "Foi detectado uma movimentação suspeita em sua residencia."
    img_data = open('frame0.jpg', 'rb').read()
    #img = "frame0.jpg"
    # setup the parameters of the message
    password = "semb2018"
    msg['From'] = "systemseg3@gmail.com"
    msg['To'] = "iago.g.vargas@gmail.com"
    msg['Subject'] = "Detalhes de Suspeita"
    
    msg.attach(MIMEImage(img_data, 'jpeg'))
    msg.attach(MIMEText(message, 'plain'))
    
    #create server
    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()
    
    # Login Credentials for sending the mail
    server.login(msg['From'], password)
    
    # send the message via the server.
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    
    server.quit()
    print ("successfully sent email to %s:" % (msg['To']))
    

def sendmessage():
    # Your Account SID from twilio.com/console
    account_sid = "AC24c1873b893995904c3db8db439f4dfb"
    # Your Auth Token from twilio.com/console
    auth_token  = "2766a642e6177645f43fb1da923e1872"
    client = Client(account_sid, auth_token)
    message = client.messages.create(
            to="+5534992248367", 
            from_="+12763789121",
            body="ATENÇÃO!!! SUA RESIDÊNCIA PODE ESTÁ EM PERIGO! Acesse seu email para mais detalhes!")
    print(message.sid)

    
def detectar():
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
    #COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    count = 0
    
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
    
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()
    # loop over the frames from the video stream
    
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # draw the prediction on the frame
                #label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                #cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                #y = startY - 15 if startY - 15 > 15 else startY + 15
                #cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
           
        # show the output frame
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if idx == 15 and confidence > 0.6:
            time.sleep(4)
            cv2.imwrite("frame%d.jpg" % count, frame)
            break
        # update the FPS counter
        fps.update()
        
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    #print (label[0:6])
    
    # do a bit of cleanup
    vs.stop()
    cv2.destroyAllWindows()
    sendmessage()
    sendemail()


detectar()