import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            h, w, c = img.shape
            lm = handLms.landmark

            x1, y1 = int(lm[4].x*w), int(lm[4].y*h)   # thumb tip
            x2, y2 = int(lm[8].x*w), int(lm[8].y*h)   # index tip

            cx, cy = (x1+x2)//2, (y1+y2)//2

            cv2.circle(img,(x1,y1),10,(255,0,0),-1)
            cv2.circle(img,(x2,y2),10,(255,0,0),-1)
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)

            length = np.hypot(x2-x1,y2-y1)

            volume = np.interp(length,[20,200],[0,100])

            # macOS volume control
            os.system(f"osascript -e 'set volume output volume {int(volume)}'")

            cv2.putText(img,f'Volume: {int(volume)}%',(40,70),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

            mp_draw.draw_landmarks(img,handLms,mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Volume Control",img)

    if cv2.waitKey(1)==27:
        break