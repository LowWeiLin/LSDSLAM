import numpy as np
import cv2

video_name = 'g_2'

cap = cv2.VideoCapture(video_name + '.avi')

frame_num = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
    	break

    frame = resized_image = cv2.resize(frame, (640, 480)) 

    cv2.imwrite(video_name + '/' + str(frame_num).zfill(6)+".png", frame)

    frame_num += 1

    #cv2.imshow('frame',gray)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

print "Done"
cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()