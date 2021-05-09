import cv2
import numpy as np
import face_recognition

# loading image
# Converting BGR to RGB

imgElon = face_recognition.load_image_file('ImageBasics/Elon musk.jpg')  # loading image
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)          # Converting BGR to RGB

imgTest = face_recognition.load_image_file('ImageBasics/Elon test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# Finding faces in our image
while True:
    faceloc = face_recognition.face_locations(imgElon)[0]   # finding face in picture
    encodeElon = face_recognition.face_encodings(imgElon)[0]  # encode the detected face
    cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)    # detected face location


# Finding faces in our test image
    facelocTest= face_recognition.face_locations(imgTest)[0]   # finding face in picture
    encodeTest = face_recognition.face_encodings(imgTest)[0]  # encode the detected face
    cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

# comparing between faces and finding measurements(distance between them)
# comparing the two images using linear encoding

    results = face_recognition.compare_faces([encodeElon],encodeTest)

# for finding the best match we will calculate the distance
    faceDis=face_recognition.face_distance([encodeElon],encodeTest)  # lower the faceDis more chances of matching
    print(results,faceDis)
    cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(10,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    cv2.imshow('Elon Musk',imgElon)         # Showing image
    cv2.imshow('Elon Test',imgTest)
    cv2.waitKey(0)
