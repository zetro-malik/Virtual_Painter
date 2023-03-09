import cv2 as cv
import mediapipe as mp 
import time


class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon=detectionCon
        self.trackingCon= trackingCon


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,1,self.detectionCon,self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils  # built in function for to draw points on hand, although the mediapipe hand solution gives as landmarkpoint but for use OPENcv draw it will take alot of time and math, instead use this 
        self.tipIds = [4,8,12,16,20]



    def findHands(self,img,draw=True):
            imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            self.result = self.hands.process(img)
            #print(result.multi_hand_landm arks)  for get landmarks of Hand
            
            if  self.result.multi_hand_landmarks:
                for handlms in  self.result.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
            return img
  
    def findPosition(self, img, handNo=0,draw=True):
        self.lmlist = []
        if self.result.multi_hand_landmarks:
           myHand= self.result.multi_hand_landmarks[handNo]
           for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,255),cv.FILLED)
              
           
            
        return self.lmlist
    
    def fingersUp(self):
        fingers = []

        if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 for fingers
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
   

def main():
    ptime = 0
    ctime = 0
    cap =cv.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = cv.flip(img,1)
       
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        img = detector.findHands(img,draw=False)
        lst = detector.findPosition(img,draw=False)
        # if len(lst) !=0:
        #     print(lst[8])

        cv.putText(img, str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv.imshow("Image",img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()