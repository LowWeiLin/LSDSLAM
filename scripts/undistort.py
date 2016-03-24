import numpy as np
import cv2

video_name = 'chess'

cap = cv2.VideoCapture(video_name + '.MP4')

frame_num = 0

'''
# GoPro studio processed
camera_matrix = np.matrix(  [[ 282.3554769,   0.           , 316.8959948 ],
                             [   0.       ,   282.37401685 , 245.18105368],
                             [   0.       ,   0.           , 1.          ]])

dist_coeffs = np.array([-0.00999921,  0.01087428,  0.00470071, -0.00068435, -0.00136128])
'''
'''
#RMS: 1.64458559933
camera_matrix = np.matrix(  [[ 887.85486526,    0.        ,  963.9280898 ],
                             [   0.        ,  885.03581399,  553.1136225 ],
                             [   0.        ,    0.        ,    1.        ]])
dist_coeffs = np.array([-0.28549717,  0.19031869,  0.00249047,  0.00037897, -0.22055651])
'''

#RMS: 0.319998257005
camera_matrix = np.matrix(  [[ 402.08205919,    0.        ,  316.8862338 ],
                             [   0.        ,  533.80324003,  215.57511076],
                             [   0.        ,    0.        ,    1.        ]])
dist_coeffs = np.array([-0.24311993, -0.08282824,  0.0023221,   0.00051102,  0.14457063])

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    resized_image = cv2.resize(frame, (640, 480)) 

    # undistort!
    corrected = cv2.undistort(resized_image, camera_matrix, dist_coeffs)



    cv2.imwrite(video_name + '_undistorted/' + str(frame_num).zfill(6)+".png", corrected)

    #cv2.imshow('frame', frame)
    #cv2.imshow('corrected', corrected)

    frame_num += 1

    #cv2.imshow('frame',gray)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

    #cv2.waitKey(0)
     
print "Done"
cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()