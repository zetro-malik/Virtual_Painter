import cv2 as cv
import mediapipe as mp 
import time


cap =cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # built in function for to draw points on hand, although the mediapipe hand solution gives as landmarkpoint but for use OPENcv draw it will take alot of time and math, instead use this 

ptime = 0
ctime = 0
cricle = []
while True:
    success, img = cap.read()
    img = cv.flip(img,1)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    result = hands.process(img)
    #print(result.multi_hand_landmarks)  for get landmarks of Hand
    
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                
                #print(id,cx, cy)
                if id ==8:
                    cricle.append([cx,cy])
                    for i in cricle:
                        cv.circle(img,(int(i[0]),int(i[1])),25,(255,0,255),cv.FILLED)
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv.putText(img, str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow("Image",img)
    cv.waitKey(1)
