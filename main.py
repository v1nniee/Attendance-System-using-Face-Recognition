import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'Training images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            person_name = entry[0]
            attendance_date = entry[1].strip()

            nameList.append(person_name)
            if name == person_name:
                current_date = datetime.now().strftime("%m/%d/%Y")
                if current_date in attendance_date:
                    return False  # Return False if attendance is already marked for the time threshold

        now = datetime.now()
        dtString = now.strftime("%m/%d/%Y,%H:%M:%S")
        f.writelines(f'\n{name},{dtString}')
    return True


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    #modify algorithm = facesCurFrame = face_recognition.face_locations(imgS, number_of_times_to_upsample=2)
    #add a new folder named  Additional Training Images

    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)


    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)

    c = cv2.waitKey(1)

    if c == 27: #press esc file to exit - https://subscription.packtpub.com/book/data/9781785283932/3/ch03lvl1sec28/accessing-the-webcam
        break
    elif c == 32: #press space bar to screenshot the new people and add to training images folder - https://www.educative.io/answers/how-to-capture-a-single-photo-with-webcam-using-opencv-in-python
        name = input("Enter your name: ")
        image_name = os.path.join(path, f"{name}.jpg")
        cv2.imwrite(image_name, img)
        print(f"Screenshot image saved as {name}.jpg")

cap.release()
cv2.destroyAllWindows()