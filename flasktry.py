from flask import Flask, render_template, request
import json
import math
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp                              # Library for image processing
from math import floor
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
import os
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/p2.html')
def plot():
    return render_template('p2.html')
@app.route('/p3.html')
def plot1():
    return render_template('p3.html')
@app.route('/p4.html')
def plot2():
    return render_template('p4.html')
@app.route('/p5.html')
def plot3():
    return render_template('p5.html')
@app.route('/p6.html')
def plot4():
    return render_template('p6.html')
@app.route('/p7.html')
def plot5():
    return render_template('p7.html')
@app.route('/p8.html')
def plot6():
    return render_template('p8.html')
@app.route('/p9.html')
def plot7():
    return render_template('p9.html')
@app.route('/prod_detail.html')
def plot8():
    return render_template('prod_detail.html')
@app.route('/product.html')
def plot9():
    return render_template('product.html')

@app.route('/pant.html')
def ploty():
    return render_template('pant.html')

#######################################################
def mdpt(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5) 

@app.route('/measurement', methods=['GET','POST'])
def measurement():
    cap = cv2.VideoCapture(0)

# Set mediapipe pose model
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Get reference image
    ref_image = cv2.imread("img.jpg")
    ref_image_height, ref_image_width, _ = ref_image.shape

    # Calculate reference shoulder width
    mp_holistic = mp.solutions.holistic
    mp_drawing_styles = mp.solutions.drawing_styles
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        ref_l_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        ref_r_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ref_shoulder_width = math.dist([ref_l_shoulder.x*ref_image_width, ref_l_shoulder.y*ref_image_height],
                                    [ref_r_shoulder.x*ref_image_width, ref_r_shoulder.y*ref_image_height])

    # Loop over frames from the video stream
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect the pose from the frame
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(frame)
            
            # Draw pose landmarks on the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Calculate the shoulder width of the user
                l_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_width = math.dist([l_shoulder.x*frame.shape[1], l_shoulder.y*frame.shape[0]],
                                        [r_shoulder.x*frame.shape[1], r_shoulder.y*frame.shape[0]])
                
                # Calculate the shirt size of the user based on the shoulder width
                shirt_size = "Not Determined"
                if shoulder_width < ref_shoulder_width * 0.9:
                    shirt_size = "S"
                elif shoulder_width < ref_shoulder_width * 1.1:
                    shirt_size = "M"
                elif shoulder_width < ref_shoulder_width * 1.3:
                    shirt_size = "L"
                else:
                    shirt_size = "XL"
                
                # Put the shirt size text on the frame
                cv2.putText(frame, "Shirt Size: " + shirt_size, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

    # Check for the 'q' key to exit
        if cv2.waitKey(1) == ord('q'):
            break

# Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')
#########################################################

@app.route('/scan', methods=['GET','POST'])
def scan():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    
    shirtFolderPath = r"C:\Users\DELL\Desktop\ipd\FitShot\Shirts"
    listShirts = os.listdir(shirtFolderPath)
    # print(listShirts)
    fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12   # this the point of shoulders
    shirtRatioHeightWidth = 581 / 440  # width line of the shirt

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        # img = cv2.flip(img,1)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            # center = bboxInfo["center"]
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[0]), cv2.IMREAD_UNCHANGED)
    
            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            print(widthOfShirt)
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentScale), int(48 * currentScale)
    
            try:
                img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except:
                pass

        cv2.imshow("Image", img)
        cv2.waitKey(1)

@app.route('/pict/<int:shirtno>', methods=['GET','POST'])
def pict(shirtno):
    
    print(shirtno)

    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    
    shirtFolderPath = r"C:\Users\DELL\Desktop\ipd\FitShot\Shirts"
    listShirts = os.listdir(shirtFolderPath)
    # print(listShirts)
    fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12   # this the point of shoulders
    shirtRatioHeightWidth = 581 / 440  # width line of the shirt

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        # img = cv2.flip(img,1)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            # center = bboxInfo["center"]
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[shirtno]), cv2.IMREAD_UNCHANGED)
    
            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            print(widthOfShirt)
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentScale), int(48 * currentScale)
    
            try:
                img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except:
                pass

        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) == ord('q'):
            break

# Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')
    
   
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)





