import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX #opencv에서 지원하는 font

id = 0

names = ['None','HwangDongJun','KimSuMin']
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 2)
        #predict에 대한 설명은 blog
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 55):
            id = names[id]
            confidence = " {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = " {0}%".format(round(100 - confidence))
        #일치 확률과 이름을 화면에 출력
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,255,0),2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (0,255,0),2)

    cv2.imshow('camera',img)
    #최대한 자주 Key를 획득할 수 있도록 wait time을 줄임
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

print("\nExisting Program.")
cap.release()
cv2.destroyAllWindows()