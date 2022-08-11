#Now since our model is trained and ready we can now begin our recognition process

import cv2 as cv
import numpy as np

#Next we grab our haar cascade file

haar_cascade=cv.CascadeClassifier('Face_Detection_Builtin/Harr_faces.xml')

#We again intialize out face recognizer

face_recognizer=cv.face.LBPHFaceRecognizer_create()

#Now we read into our yml file

face_recognizer.read("face_Trained.yml")

#Next we grab our peoples list

people=["shah rukh khan","Stephen hawking"]

#And thats it we have transfered our model

#Next we grab our image to identify

img_init=cv.imread("Face_Recognition_Builtin/Faces/Stephen Hawking/124790-chin-cheek-human-nose-eyebrow-1865x2500.jpg")

img=cv.resize(img_init,(500,500),interpolation=cv.INTER_AREA)

cv.imshow("img",img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#Now we can detect our face

faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

for (x,y,w,h)in faces_rect:
    face_roi=gray[y:y+h,x:x+w] #We again get our face region of interest

    #Next we use the predict function of the LBPH face recognizer which will retrn two values
    #1)The label(i.e the person in the image) 2)The confidemce percentage(This shows that how confident the model is on its prediction)

    label,confidence =face_recognizer.predict(face_roi)

    print(f"Predicted with a confidence of:{confidence}")

    #We put in the name of the person in the image
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)

    #We can also create a rectangle over the detected face

    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("Detected image",img)
cv.waitKey(0)


