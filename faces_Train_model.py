#Next we look at face recognition.For face recognition we have to create a mini deep learning model,but here we will be using opencv 
# built in face recognizer so we dont have to create a model from scratch but we do have to train it with images

#We have taken 15 pictures each of our subjects to train our model

#We will be using pythons os module in this.It is similar to the os module in node.js which helps us in navigating through different
# components of the os in this case we will be navigating through files.More specifically our images

import os  
import cv2 as cv
import numpy as np

#Next we create a list of the people we want to recognize and we will loop over this list,Go into each folder with the given name in the 
#list and then look at each image in the list


people=["shah rukh khan","Stephen hawking"]

#We also store the address of the directory in which the photos are present in a variable

Dir= r"C:\Users\Shirshak\Desktop\Open CV\Face_Recognition_Builtin\Faces"

#Before recognizing an image we first need to detect whether an image exist or not and so we will need the haar cascade classifier 

haar_cascade=cv.CascadeClassifier('Face_Detection_Builtin/Harr_faces.xml')

 #Now we create two empty list one for storing our list of faces and another for keeping a track of whose image is it

faces=[]
labels=[]

#Next we create a function to train our model

def create_train():
    #Now we iterate over the list
    for person in people:
        #Next we get the path to each image
        #for this we use the os.path.join function which will join the base directory path with the folder of our subjects

        path=os.path.join(Dir,person)

        #Next we store the index of appereance of each person in our folder we use the .index function for this

        label=people.index(person)
        #Now we have iterated over the main folder now we go inside the main folder qand iterate over each image

        #We use the listdir function of the os module for this,This will retrun all the path of all the documents present in the 
        # file passed to it.So since here we are passing our subjects image folder path,it will return a list  of all the path of 
        # all the images in that folder
         
        for img in os.listdir(path):
            #Next we join the file path of each of our subjects imaqge folder to the image path
            img_path=os.path.join(path,img)

            #Next we read each of the image in each folder
            img_array=cv.imread(img_path)

            #Now comes our face detection part
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

            #Now we since we have detected the face we want to focus only on the face and nowhere else this is what we call the 
            #Region of interest

            #For this we will again loop over the coordinates of our rectangle given by the detectMultiScale function

            for(x,y,w,h) in faces_rect:
                #To get our region of interest we specify the area we want o focus over in the image by passing the range of x and y 
                # coordinates 

                faces_roi=gray[y:y+h,x:x+w]

                #now since we have gotten our face we can append it to our face list

                faces.append(faces_roi)
                labels.append(label)

create_train()

print("Training Finished-----------")
print(f"Number of images/faces:{len(faces)}")

#Now our trainin model is complete we can move on to the face recognition

#For that first we need to convert our faces list and our labels ilist into a numpy array

#The reson for this is open cv needs the color height and width values of an image to recognize it and if it is a color image
# it also needs in the color channels and converting to numpy array provides these values

#you can go the following link to read more
#https://stackoverflow.com/questions/56204630/what-is-the-need-of-converting-an-image-into-numpy-array 

faces=np.array(faces,dtype="object")
labels=np.array(labels)

#Next we get our face recognizer,we use the LBPH face recognizer for this which is opencv's built in recognizer

face_recognizer=cv.face.LBPHFaceRecognizer_create()

#Now we need to train our recognizer on the faces and the labels list
face_recognizer.train(faces,labels)

#Next we save our face recognizer values and also the faces and the labels value so that when we try to use this face recognizer
#in another project we dont have to repeat the process

face_recognizer.save('face_trained.yml')
np.save('faces.npy',faces)
np.save('labels.npy',labels)
