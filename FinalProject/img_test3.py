import numpy as np
import cv2 as cv
import dlib
import os

# 얼굴 path
faceCascade = cv.CascadeClassifier("D:/python/OCV/cascades/haarcascade_frontalface_alt.xml")
predictor = dlib.shape_predictor("D:/jb_python/self_study/210727/shape_predictor_68_face_landmarks.dat")

protoFile_coco = "D:/Program Files/openpose-master/models/pose/coco/pose_deploy_linevec.prototxt"
weightsFile_coco = "D:/Program Files/openpose-master/models/pose/coco/pose_iter_440000.caffemodel"

JAWLINE_POINTS = list(range(0, 17))
BOTH_EYEBROW_POINTS = list(range(17,27))
NOSE_POINTS = list(range(27, 36))
#RIGHT_EYE_POINTS = list(range(36, 42))
#LEFT_EYE_POINTS = list(range(42, 48))
BOTH_EYES_POINTS = list(range(36,48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

 
def output_keypoints(frame, net, BODY_PARTS):
    global points
    # 얼굴 찾음
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100,100), flags=cv.CASCADE_SCALE_IMAGE)
    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward() # shape : (1, 78, 46, 46)

    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    # 해당 신체부위 신뢰도 얻음.
    probMap1 = out[0, 1, :, :]
    probMap2 = out[0, 2, :, :]
    probMap3 = out[0, 5, :, :]

    # global 최대값 찾기
    minVal1, prob1, minLoc1, point1 = cv.minMaxLoc(probMap1)
    minVal2, prob2, minLoc2, point2 = cv.minMaxLoc(probMap2)
    minVal3, prob3, minLoc3, point3 = cv.minMaxLoc(probMap3)
    
    # 원래 이미지에 맞게 점 위치 변경
    x1 = (frame_width * point1[0]) / out_width
    y1 = (frame_height * point1[1]) / out_height
    x2 = (frame_width * point2[0]) / out_width
    y2 = (frame_height * point2[1]) / out_height
    x3 = (frame_width * point3[0]) / out_width
    y3 = (frame_height * point3[1]) / out_height

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로   
    # 목
    if prob1 > 0.1 :    
        cv.circle(frame, (int(x1), int(y1)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(1), (int(x1), int(y1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x1), int(y1)))

    else:  # [not pointed]
        points.append(None)
        #print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    # 오른쪽 어깨
    if prob2 > 0.1 :
        cv.circle(frame, (int(x2), int(y2)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(2), (int(x2), int(y2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x2), int(y2)))
    else:  # [not pointed]
        points.append(None)

    # 왼쪽 어깨
    if prob3 > 0.1 :    
        cv.circle(frame, (int(x3), int(y3)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(3), (int(x3), int(y3)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x3), int(y3)))
    else:  # [not pointed]
        points.append(None)

    for (x,y,w,h) in faces:
        # opencv 이미지 > dlib용 사각형 변환
        dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        # 랜드마크 포인트 지정
        landmarks = np.matrix([[p.x,p.y] for p in predictor(frame, dlib_rect).parts()])
        
        # matrix 분리
        eyes0 = (int(landmarks[BOTH_EYES_POINTS[0], 0]), int(landmarks[BOTH_EYES_POINTS[0], 1]))
        eyes8 = (int(landmarks[BOTH_EYES_POINTS[9], 0]), int(landmarks[BOTH_EYES_POINTS[9], 1]))
        
        # 양쪽 눈꼬리
        points.append(eyes0)
        points.append(eyes8)

    # 원하는 부위 출력
    landmarks_display = points[3:]

    # 포인트 출력
    for idx, point in enumerate(landmarks_display):
        pos = (point[0], point[1])
        cv.circle(frame, pos, 3, color=(0, 255, 255), thickness=-1)
        cv.putText(frame, "eye{}".format(idx), pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)

    return frame

def output_keypoints_with_lines(frame, POSE_PAIRS):
    try:
        # 양쪽 어깨 접선
        if points[1] and points[2]:
            cv.line(frame, points[1], points[2], (0, 255, 0), 3) 
        elif points[0] and points[2]:
            cv.line(frame, points[0], points[2], (0, 255, 0), 3) 
        else:
            cv.line(frame, points[0], points[1], (0, 255, 0), 3) 
        try:
            cen_eyes_X = (points[3][0] + points[4][0]) // 2
            cen_eyes_Y = (points[3][1] + points[4][1]) // 2

            cen_sholder_X = (points[1][0] + points[2][0]) // 2
            cen_sholder_Y = (points[1][1] + points[2][1]) // 2

            cen_eyes = (cen_eyes_X, cen_eyes_Y)
            cen_sholder = (cen_sholder_X, cen_sholder_Y)
            
            if cen_eyes and cen_sholder:
                cv.line(frame, cen_eyes, cen_sholder, (0, 0, 255), 3) 
        except:
            cen_eyes_X = (points[3][0] + points[4][0]) // 2
            cen_eyes_Y = (points[3][1] + points[4][1]) // 2
            cen_eyes = (cen_eyes_X, cen_eyes_Y)
            if cen_eyes and points[0]:
                cv.line(frame, cen_eyes, points[0], (0, 0, 255), 3) 
        
        # 눈꼬리 접선
        if points[3] and points[4]:
            cv.line(frame, points[3], points[4], (0, 255, 0), 3)
    except (IndexError):
        pass
    except (TypeError):
        pass
    return frame

def output_keypoints_with_lines_video(proto_file, weights_file, BODY_PARTS, POSE_PAIRS):

    # 네트워크 불러오기
    net = cv.dnn.readNetFromCaffe(proto_file, weights_file)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    load_img = "./dataset/corr-samples/"
    save_img = "./dataset/corr_posture/correct-"
    # load_img = "./dataset/forw-samples/"
    # save_img = "./dataset/forw_posture/forward-"
    for img in os.listdir(load_img):
        
        aa = load_img + img
        cap = cv.imread(aa, cv.IMREAD_COLOR)

        cap = output_keypoints(frame=cap, net=net, BODY_PARTS=BODY_PARTS)
        cap = output_keypoints_with_lines(frame=cap, POSE_PAIRS=POSE_PAIRS)
        cv.imwrite(save_img+img, cap)

# 키포인트를 저장할 빈 리스트
points = []

output_keypoints_with_lines_video(proto_file=protoFile_coco, weights_file=weightsFile_coco, BODY_PARTS=BODY_PARTS_COCO, POSE_PAIRS=POSE_PAIRS_COCO)

cv.waitKey(0) # esc 입력시 종료
cv.destroyAllWindows()