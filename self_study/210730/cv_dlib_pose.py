import numpy as np
import cv2 as cv
import dlib

faceCascade = cv.CascadeClassifier("D:/python/OCV/cascades/haarcascade_frontalface_alt.xml")
predictor = dlib.shape_predictor("C:/Users/marbi/0.python/self_study/210727/shape_predictor_68_face_landmarks.dat")

JAWLINE_POINTS = list(range(0, 17))
# RIGHT_EYEBROW_POINTS = list(range(17, 22))
# LEFT_EYEBROW_POINTS = list(range(22, 27))
BOTH_EYEBROW_POINTS = list(range(17,27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
# 각 파일 path
protoFile = "D:/Program Files/openpose-master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "D:/Program Files/openpose-master/models/pose/mpi/pose_iter_160000.caffemodel"
 
# 위의 path에 있는 network 불러오기
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

def detect(gray, frame):
    # 얼굴 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100,100), flags=cv.CASCADE_SCALE_IMAGE)
    want_point = []
    # 랜드마크 찾음
    for (x,y,w,h) in faces:
        # opencv 이미지 > dlib용 사각형 변환
        dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        # 랜드마크 포인트 지정
        landmarks = np.matrix([[p.x,p.y] for p in predictor(frame, dlib_rect).parts()])
        # 원하는 포인트 넣음 (현재 전부)
        #landmarks_display = landmarks[0:68]

        # 양 턱 시작, 턱 끝, 코 끝
        want_point.append(landmarks[JAWLINE_POINTS[0]])
        want_point.append(landmarks[JAWLINE_POINTS[8]])
        want_point.append(landmarks[JAWLINE_POINTS[16]])
        want_point.append(landmarks[NOSE_POINTS[3]])
        # 원하는 부위 출력
        landmarks_display = want_point

        # 포인트 출력
        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv.circle(frame, pos, 10, color=(0, 255, 255), thickness=-1)

    return frame

# 웹캠 이미지 가져오기
video_capture = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    # 웹캠 이미지 프레임 화
    _, frame = video_capture.read()
 
    
    frameHeight, frameWidth, _ = frame.shape
 
    # network에 넣기위해 전처리
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (frameWidth, frameHeight), (0, 0, 0), swapRB=False, crop=False)
    
    # network에 넣어주기
    net.setInput(inpBlob)

    # 결과 받아오기
    output = net.forward()

    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    H = output.shape[2]
    W = output.shape[3]
    print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

    # 키포인트 검출시 이미지에 그려줌
    points = []
    # for i in range(2):
    # 해당 신체부위 신뢰도 얻음.
    probMap1 = output[0, 2, :, :]
    probMap2 = output[0, 5, :, :]

    # global 최대값 찾기
    minVal1, prob1, minLoc1, point1 = cv.minMaxLoc(probMap1)
    minVal2, prob2, minLoc2, point2 = cv.minMaxLoc(probMap2)
    # 원래 이미지에 맞게 점 위치 변경
    x1 = (frameWidth * point1[0]) / W
    y1 = (frameHeight * point1[1]) / H
    x2 = (frameWidth * point2[0]) / W
    y2 = (frameHeight * point2[1]) / H

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
    if prob1 > 0.1 :    
        cv.circle(frame, (int(x1), int(y1)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(2), (int(x1), int(y1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x1), int(y1)))
    else :
        points.append(None)
    if prob2 > 0.1 :
        cv.circle(frame, (int(x2), int(y2)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(5), (int(x2), int(y2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x2), int(y2)))
    else :
        points.append(None)

    if points[0] and points[1]:
        cv.line(frame, points[0], points[1], (0, 255, 0), 2)
    # q 누르면 종료
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
       # 좌우 반전
    frame = cv.flip(frame,1)
    # 그레이스케일 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 얼굴 눈 찾기
    canvas = detect(gray, frame)

    # 이미지 보여주기
    cv.imshow("haha", canvas)

video_capture.release()
cv.destroyAllWindows()