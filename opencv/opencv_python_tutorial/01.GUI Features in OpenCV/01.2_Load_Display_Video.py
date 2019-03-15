## Pacakages
import numpy as np
import cv2

## cap is VideoCapture Object, its argument can be
## either the device index or the name of a video file.
## Device Index : (0 or -1) , 1 and so on.
cap = cv2.VideoCapture(0)

while(True):
    ## Capture frame-by-frame
    #cap.read() return true/false
    ret, frame = cap.read()

    ## Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

## When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
