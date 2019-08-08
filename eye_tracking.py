# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 15

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True

vs = VideoStream(src=0).start()
#vs = FileVideoStream(args["video"]).start()

fileStream = False
time.sleep(1.0)

# opening windows to display images and moving them to right positions
cv2.namedWindow('Frame')
cv2.moveWindow('Frame',0,0)

cv2.namedWindow('Left Eye')
cv2.moveWindow('Left Eye',550,0)
cv2.namedWindow('Equalized Left Eye')
cv2.moveWindow('Equalized Left Eye',550,300)

cv2.namedWindow('Right Eye')
cv2.moveWindow('Right Eye',950,0)
cv2.namedWindow('Equalized Right Eye')
cv2.moveWindow('Equalized Right Eye',950,300)

img_no = 0
phone_call_done = 0
flashlight_done = 0
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# cropping out the left and the right eye
		(x, y, w, h) = cv2.boundingRect(np.array([shape[lStart:lEnd]]))
		left_eye_image = frame[y:y + h, x:x + w]
		left_eye_image = left_eye_image.copy()
		#left_eye_image = imutils.resize(left_eye_image, width=100)
		
		(x, y, w, h) = cv2.boundingRect(np.array([shape[rStart:rEnd]]))
		right_eye_image = frame[y:y + h, x:x + w]
		right_eye_image = right_eye_image.copy()
		#right_eye_image = imutils.resize(right_eye_image, width=100)

		#Finding centre of left and right eye and
		# drawing it on their images
		M_left = cv2.moments(leftEye)
                cX_left = int(M_left["m10"] / M_left["m00"])
                cY_left = int(M_left["m01"] / M_left["m00"])
                M_right = cv2.moments(rightEye)
                cX_right = int(M_right["m10"] / M_right["m00"])
                cY_right = int(M_right["m01"] / M_right["m00"])
                cv2.circle(frame, (cX_left, cY_left), 1, (0, 0, 255), -1)
                cv2.circle(frame, (cX_right, cY_right), 1, (0, 0, 255), -1)

                # Finding Hough Circles in left and right eye images
                gray_left = cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2GRAY)
                equ_left = cv2.equalizeHist(gray_left)
                gray_right = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2GRAY)
                equ_right = cv2.equalizeHist(gray_right)
                #circles_left = cv2.HoughCircles(equ_left,cv2.HOUGH_GRADIENT,1,gray_left.shape[0]/8,param1=250,param2=15,minRadius=gray_left.shape[1]/8,maxRadius=gray_left.shape[0]/3)
                circles_left = cv2.HoughCircles(equ_left,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
                if isinstance(circles_left, np.ndarray):
                        circles_left = np.uint16(np.around(circles_left))
                        mean_values_left = []
                        for i in circles_left[0,:]:
##                                width = equ_left.shape[0]
##                                height = equ_left.shape[1]
##                                circle_img = np.zeros((height,width), np.uint8)
##                                cv2.circle(circle_img,(i[0],i[1]),i[2],1,thickness=-1)
##                                masked_data = cv2.bitwise_and(gray_left, gray_left, mask=circle_img)
##                                mean_val = cv2.mean(equ_left,mask = masked_data)
##                                mean_values_left.append(mean_val)
##                        max_index_left = [i for i, j in enumerate(mean_values_left) if j == min(mean_values_left)]
##                        eyeball_x_left = circles_left[0,:][max_index_left][0][0]
##                        eyeball_y_left = circles_left[0,:][max_index_left][0][1]
##                        eyeball_r_left = circles_left[0,:][max_index_left][0][2]
                                cv2.circle(left_eye_image,(i[0],i[1]),i[2],(0,255,0),2)
                #circles_right = cv2.HoughCircles(equ_right,cv2.HOUGH_GRADIENT,1,gray_right.shape[0]/8,param1=250,param2=15,minRadius=gray_right.shape[1]/8,maxRadius=gray_right.shape[0]/3)
                circles_right = cv2.HoughCircles(equ_right,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
                if isinstance(circles_right, np.ndarray):
                        circles_right = np.uint16(np.around(circles_right))
                        mean_values_right = []
                        for i in circles_right[0,:]:
##                                width = equ_right.shape[0]
##                                height = equ_right.shape[1]
##                                circle_img = np.zeros((height,width), np.uint8)
##                                cv2.circle(circle_img,(i[0],i[1]),i[2],1,thickness=-1)
##                                masked_data = cv2.bitwise_and(gray_right, gray_right, mask=circle_img)
##                                mean_val = cv2.mean(equ_right,mask = masked_data)
##                                mean_values_right.append(mean_val)
##                        max_index_right = [i for i, j in enumerate(mean_values_right) if j == min(mean_values_right)]
##                        eyeball_x_right = circles_right[0,:][max_index_right][0][0]
##                        eyeball_y_right = circles_right[0,:][max_index_right][0][1]
##                        eyeball_r_right = circles_right[0,:][max_index_right][0][2]
                                cv2.circle(right_eye_image,(i[0],i[1]),i[2],(0,255,0),2)

                        

##		# thresholding eye images
##		THRESH_CONST = 20
##		gray_left = cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2GRAY)
##		gray_right = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2GRAY)
##		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##		ret,thresh_left = cv2.threshold(gray_left,THRESH_CONST,255,cv2.THRESH_BINARY_INV)
##		ret,thresh_right = cv2.threshold(gray_right,THRESH_CONST,255,cv2.THRESH_BINARY_INV)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			# reset the eye frame counter
			COUNTER = 0


		# Home Automation

		
		if TOTAL == 1:
                        pass
                elif TOTAL == 2 and phone_call_done == 0:
                        os.system("python /home/apoorv/Desktop/blinkNdo/phone_call_updated.py")
                        phone_call_done = 1
                elif TOTAL == 3 and flashlight_done == 0:
                        os.system("adb shell input tap 540 960")
                        flashlight_done = 1
                elif TOTAL == 4 and flashlight_done == 1:
                        os.system("adb shell input tap 540 960")
                        flashlight_done = 0
		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# show the left eye
                cv2.imshow("Left Eye", left_eye_image)
                # show the right eye
                cv2.imshow("Right Eye", right_eye_image)
                # show the thresholded left eye
                cv2.imshow("Equalized Left Eye", equ_left)
                # show the thresholded right eye
                cv2.imshow("Equalized Right Eye", equ_right)
 
	# show the frame
	cv2.imshow("Frame", frame)
	
	key = cv2.waitKey(1) & 0xFF
##        if key == ord("1"):
##                cv2.imwrite('1/5_' + str(img_no) + '.jpg',left_eye_image)
##                img_no = img_no + 1
##                cv2.imwrite('1/5_' + str(img_no) + '.jpg',right_eye_image)
##                img_no = img_no + 1
##        elif key == ord("2"):
##                cv2.imwrite('2/5_' + str(img_no) + '.jpg',left_eye_image)
##                img_no = img_no + 1
##                cv2.imwrite('2/5_' + str(img_no) + '.jpg',right_eye_image)
##                img_no = img_no + 1
##        elif key == ord("3"):
##                cv2.imwrite('3/5_' + str(img_no) + '.jpg',left_eye_image)
##                img_no = img_no + 1
##                cv2.imwrite('3/5_' + str(img_no) + '.jpg',right_eye_image)
##                img_no = img_no + 1
##        elif key == ord("4"):
##                cv2.imwrite('4/5_' + str(img_no) + '.jpg',left_eye_image)
##                img_no = img_no + 1
##                cv2.imwrite('4/5_' + str(img_no) + '.jpg',right_eye_image)
##                img_no = img_no + 1
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
