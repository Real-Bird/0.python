import numpy as np
import cv2
import dlib

faceCascade = cv2.CascadeClassifier("D:/python/OCV/cascades/haarcascade_frontalface_alt.xml")
predictor = dlib.shape_predictor("C:/Users/marbi/0.python/self_study/210727/shape_predictor_68_face_landmarks.dat")

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

def detect(gray, frame):
    # 얼굴 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)

    # 랜드마크 찾음
    for (x,y,w,h) in faces:
        # opencv 이미지 > dlib용 사각형 변환
        dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        # 랜드마크 포인트 지정
        landmarks = np.matrix([[p.x,p.y] for p in predictor(frame, dlib_rect).parts()])
        # 원하는 포인트 넣음 (현재 전부)
        landmarks_display = landmarks[0:68]
        # 눈만
        #landmarks_display = landmarks[RIGHT_EYE_POINTS, LEFT_EYE_POINTS]

        # 포인트 출력
        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

    return frame

# 웹캠 이미지 가져오기
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # 웹캠 이미지 프레임 화
    _, frame = video_capture.read()
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 눈 찾기
    canvas = detect(gray, frame)
    # 이미지 보여주기
    cv2.imshow("haha", canvas)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()