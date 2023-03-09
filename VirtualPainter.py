import cv2 as cv 
import mediapipe
import numpy as np
import os
import HandTrackingModule as htm



brushThickness =15
eraserThickness=50

folderPath = os.path.join("Resources")
overlaylist = []
for imPath in os.listdir(folderPath):
    image = cv.imread( os.path.join(folderPath+"/"+imPath))
    image = cv.resize(image,(1280,120), interpolation=cv.INTER_AREA)
    overlaylist.append(image)

print(len(overlaylist))

header = overlaylist[0]
drawColor = (255,0,255)

imgCanvas = np.zeros((720,1280,3),np.uint8)

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.HandDetector(detectionCon=0.85)
xp,yp = 0,0
while True:
    # import the image
    success, img =cap.read()
    img = cv.flip(img,1)

    # find hand landmark
    img = detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    
    if len(lmlist) !=0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

        

        # check which fingers are up

        fingers=detector.fingersUp()

        # if Selection mode, Then Select
        if fingers[1] and fingers[2]:
            xp,yp =0,0
            if y1 < 125:
                if 250<x1<450:
                    header = overlaylist[0]
                    drawColor =  (255,0,255)
                elif 550<x1<750:
                    header = overlaylist[1]
                    drawColor = (255,0,0)
                elif 800<x1<950:
                    header = overlaylist[2]
                    drawColor = (0,255,0)
                elif 1050<x1<1200:
                    header = overlaylist[3]
                    drawColor = (0,0,0)

            cv.rectangle(img, (x1,y1-25),(x2,y2+25),drawColor,cv.FILLED)


        # if drawing mode if index finger is up
        if fingers[1] and fingers[2]==False:
            cv.circle(img, (x1,y1),15,drawColor,cv.FILLED)
            
            if (xp==0 and yp==0):
                xp,yp = x1,y1
            if drawColor == (0,0,0):
                cv.line(img, (xp,yp),(x1,y1),drawColor,eraserThickness)
                cv.line(imgCanvas, (xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv.line(img, (xp,yp),(x1,y1),drawColor,brushThickness)
                cv.line(imgCanvas, (xp,yp),(x1,y1),drawColor,brushThickness)

            xp,yp = x1,y1


    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _,imgInv = cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(imgInv,img)
    img = cv.bitwise_or(img,imgCanvas)
    # setting header image
    img [0:120,0:1280]= header


    cv.imshow("Image",img)
   
 

    cv.waitKey(1)
