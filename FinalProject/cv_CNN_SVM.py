import numpy as np
import cv2 as cv
import imutils
import mediapipe as mp
import joblib

pb = "D:/jb_python/FinalProject/zz/keras_model_210830.pb"
face_cascade = cv.CascadeClassifier('D:/jb_python/FinalProject/tn_model/haarcascade_frontalface_default.xml')
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pkl_file = 'D:/jb_python/FinalProject/SVM(c=1,gamma=1).pkl'
estimator = joblib.load(pkl_file)

def face_detect(frame, cascade, net):

    frame = imutils.resize(frame, width=224, height=224)

    faces_cnt = 0
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if faces_cnt != len(faces) :
            faces_cnt = len(faces)
            if faces_cnt != 0 :
                print("현재 검출된 얼굴 수 : ", str(faces_cnt))
                frame = t_predict(frame, net)
    else:
        cv.putText(frame, "not found face.", (50, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv.LINE_AA)

    # 검출된 안면에 사각형 그리기
    # cv2.rectangle(영상이미지, (x1, y1), (x2, y2), (B, G, R), 두깨, 선형타입)
    # (X1, Y1) 좌측 상단 모서리, (X2, Y2) 우측 하단 모서리.
    if len(faces) :
        for  x, y, w, h in faces :
            cv.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2, cv.LINE_4)

    return frame

def t_predict(frame, net):
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        # 프레임 resize
        frame = imutils.resize(frame, width=224, height=224)

        # 입력 이미지의 사이즈 정의
        image_height = 224
        image_width = 224

        # 네트워크에 넣기 위한 전처리
        input_blob = cv.dnn.blobFromImage(frame, 1.0/255, (image_width, image_height), (0, 0, 0), swapRB=True, crop=False)

        # 전처리된 blob 네트워크에 입력
        net.setInput(input_blob)

        # 결과 받아오기
        out = net.forward() # shape : (1, 1)

        # SVM
        results = pose.process(frame)
        if results.pose_landmarks is None:
            cv2.putText(image, "No land mark", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, lineType=cv2.LINE_AA)
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

        test_data = [[eye_length, shoulder_length, ES_cen, LES, RES, NS]]
        pred = estimator.predict(test_data)

        percentage = float(out[0, 0]) * 100
        # bounding box 레이블 설정
        label = "co: " if (out[0, 0] < 0.5) and (pred == 0) else "fo: "
        
        cv.rectangle(frame, (0, 0), (120, 40), (255, 255, 255), -1)
        # bounding box 출력
        if out[0, 0]  < 0.5 and pred == 0:  
            cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, lineType=cv.LINE_AA)
            cv.putText(frame, "{:0.1f}".format(percentage), (50, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, lineType=cv.LINE_AA)
        elif (out[0, 0] < 0.5 and pred == 1) or (out[0, 0] > 0.5 and pred == 0):
            cv.putText(frame, "Warning!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 0), 1, lineType=cv.LINE_AA)
        else:
            cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv.LINE_AA)
            cv.putText(frame, "{:0.1f}".format(percentage), (50, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv.LINE_AA)
        

        return frame

def output_video(pb):

    # 네트워크 불러오기
    net = cv.dnn.readNetFromTensorflow(pb)

    # openCV CUDA 없을 경우 주석 처리
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv.VideoCapture(0)
    
    while True:
        now_frame = capture.get(cv.CAP_PROP_POS_FRAMES)
        total_frame = capture.get(cv.CAP_PROP_FRAME_COUNT)
        
        if now_frame == total_frame:
            break

        ret, frame = capture.read()

        frame = cv.flip(frame,1)

        frame = face_detect(frame, face_cascade, net)

        frame = imutils.resize(frame, width=640, height=480)

        cv.imshow("Output_Video", frame)
        
        

        if cv.waitKey(1) == 27:  # esc 입력시 종료
            break

    capture.release()
    cv.destroyAllWindows()

output_video(pb=pb)