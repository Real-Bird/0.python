# Media PipePose를 위한 import
import numpy as np
import sys
import cv2
import mediapipe as mp
import joblib

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pkl_file = 'D:/jb_python/FinalProject/SVM(randomforest).pkl'
estimator = joblib.load(pkl_file)

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
     
        success, image = cap.read()
 
        image_height, image_width, _ = image.shape
     
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
   
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        results = pose.process(image)
  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  

        if results.pose_landmarks is None:
            cv2.putText(image, "No land mark", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            continue
    
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
        left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        eye_length = (abs(left_eye.x - right_eye.x))
        eye_x_centre = (left_eye.x + right_eye.x) / 2
        eye_y_centre = (left_eye.y + right_eye.y) / 2
        eye_centre = (eye_x_centre ,eye_y_centre)
        
        shoulder_length = (abs(left_shoulder.x - right_shoulder.x))
        shoulder_x_centre = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_y_centre = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_centre = (shoulder_x_centre ,shoulder_y_centre)

        ES_cen = abs(eye_centre[1] - shoulder_centre[1])
        LES = abs(left_eye.y - left_shoulder.y)
        RES = abs(right_eye.y - right_shoulder.y)
        NS = abs(nose.y - shoulder_centre[1])

        test_data = [[eye_length, shoulder_length, ES_cen, LES, RES, NS]]
        pred = estimator.predict(test_data)
        print(pred)
 
        cv2.rectangle(image, (0, 0), (120, 40), (255, 255, 255), -1)
        label = "co: " if pred == 0 else "fo: "
       
        if (pred == 0):
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(image, "{:0.1f}".format(int(pred)), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        else:
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(image, "{:0.1f}".format(int(pred)), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    
        cv2.imshow("Stop the TurtleNeck", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break        
    cap.release()
    cv2.destroyAllWindows()
