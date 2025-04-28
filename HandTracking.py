import cv2
import mediapipe as mp
import time
import math
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Camera setup
cap = cv2.VideoCapture(0)  

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# FPS tracking
pTime = 0

# Volume control setup
devices = AudioUtilities.GetSpeakers() 
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Coordinate storage
coordinatePair = [0, 0, 0, 0, 0, 0, 0, 0]  # [x1, y1, x2, y2, x3, y3, x4, y4]
last_valid_volume = None  # Store last valid volume level

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 8:  # Index Finger Tip
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
                    coordinatePair[0], coordinatePair[1] = cx, cy

                if id == 4:  # Thumb Tip
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
                    coordinatePair[2], coordinatePair[3] = cx, cy

                if id == 6:  # Middle joint of index finger
                    coordinatePair[4], coordinatePair[5] = cx, cy

                if id == 3:  # Base of thumb
                    coordinatePair[6], coordinatePair[7] = cx, cy

            cv2.line(img, (coordinatePair[0], coordinatePair[1]), (coordinatePair[2], coordinatePair[3]), (255, 0, 0), 3)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # Calculate distances
        distance1 = math.dist((coordinatePair[0], coordinatePair[1]), (coordinatePair[2], coordinatePair[3]))
        distance2 = math.dist((coordinatePair[4], coordinatePair[5]), (coordinatePair[6], coordinatePair[7]))

        #if distance2 > 0:  # Avoid division by zero
            #print(distance1 / distance2)

        # Map distance to volume
        vol = np.interp(distance1/distance2, [0.27, 1.5], [minVol, maxVol])

        # Update volume if valid hand is detected
        volume.SetMasterVolumeLevel(vol, None)
        last_valid_volume = vol  # Save the last valid volume level
        cv2.putText(img, f'Volume: {round(volume.GetMasterVolumeLevelScalar() * 100)}%', (10, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        # If no hand is detected, do nothing (retain last valid volume)
        pass

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
