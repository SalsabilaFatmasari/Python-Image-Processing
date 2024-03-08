import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime


path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    currentImage = cv2.imread(f'{path}/{cl}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtStr = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtStr}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success,  image = cap.read()
    imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            Y1, X2, Y2, X1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(image, (x1 * 4, y1 * 4), (x2 * 4, y2 * 4), (0, 255, 0), 2)  # Perbaiki lokasi kotak hijau
            cv2.rectangle(image, (x1 * 4, y2 * 4 - 35), (x2 * 4, y2 * 4), (0, 255, 0), cv2.FILLED)  # Perbaiki lokasi kotak hijau
            cv2.putText(image, name, (x1 * 4 + 6, y2 * 4 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),2)  # Perbaiki lokasi text
            markAttendance(name)

    cv2.imshow('Webcam', image)
    cv2.waitKey(1)
