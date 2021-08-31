import cv2
import mediapipe as mp
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
f1 = open('D:/jb_python/FinalProject/dataset/mp_forw.csv','w', newline='')
wr1 = csv.writer(f1)
wr1.writerow(['E_len', 'S_len', 'ESC', 'LES', 'RES', 'NS'])

load_img = "D:/jb_python/FinalProject/dataset/forw_0828/"


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    print("이미지 로드 중")
    count = 1
    for img in os.listdir(load_img):
        aa = load_img + img
        image = cv2.imread(aa, cv2.IMREAD_COLOR)

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks is None:
            pass

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
        wr1.writerow([eye_length, shoulder_length, ES_cen, LES, RES, NS])

        print("{}번째 이미지 로드 완료".format(count))
        count += 1

        # print("EC : {}, SC : {}, ESC : {}, LES : {}, RES : {}, NS : {}".format(eye_centre, shoulder_centre, ES_cen, LES, RES, NS))

        # cv2.imshow('NinjaTurtle(ver.2.0)', image)
      
cv2.waitKey()
cv2.destroyAllWindows()