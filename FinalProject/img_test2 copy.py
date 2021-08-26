import numpy as np
import cv2 as cv
import os
import dlib

# 얼굴 path
faceCascade = cv.CascadeClassifier("D:/python/OCV/cascades/haarcascade_frontalface_alt.xml")
predictor = dlib.shape_predictor("D:/jb_python/self_study/210727/shape_predictor_68_face_landmarks.dat")

protoFile_coco = "D:/Program Files/openpose-master/models/pose/coco/pose_deploy_linevec.prototxt"
weightsFile_coco = "D:/Program Files/openpose-master/models/pose/coco/pose_iter_440000.caffemodel"

JAWLINE_POINTS = list(range(0, 17))
BOTH_EYEBROW_POINTS = list(range(17,27))
NOSE_POINTS = list(range(27, 36))
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
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)

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
    probMap1 = out[0, 0, :, :]
    probMap2 = out[0, 1, :, :]
    probMap3 = out[0, 2, :, :]
    probMap4 = out[0, 5, :, :]
    probMap5 = out[0, 14, :, :]
    probMap6 = out[0, 15, :, :]

    # global 최대값 찾기
    minVal1, prob1, minLoc1, point1 = cv.minMaxLoc(probMap1)
    minVal2, prob2, minLoc2, point2 = cv.minMaxLoc(probMap2)
    minVal3, prob3, minLoc3, point3 = cv.minMaxLoc(probMap3)
    minVal4, prob4, minLoc4, point4 = cv.minMaxLoc(probMap4)
    minVal5, prob5, minLoc5, point5 = cv.minMaxLoc(probMap5)
    minVal6, prob6, minLoc6, point6 = cv.minMaxLoc(probMap6)
    
    # 원래 이미지에 맞게 점 위치 변경
    x1 = (frame_width * point1[0]) / out_width
    y1 = (frame_height * point1[1]) / out_height
    x2 = (frame_width * point2[0]) / out_width
    y2 = (frame_height * point2[1]) / out_height
    x3 = (frame_width * point3[0]) / out_width
    y3 = (frame_height * point3[1]) / out_height
    x4 = (frame_width * point4[0]) / out_width
    y4 = (frame_height * point4[1]) / out_height
    x5 = (frame_width * point5[0]) / out_width
    y5 = (frame_height * point5[1]) / out_height
    x6 = (frame_width * point6[0]) / out_width
    y6 = (frame_height * point6[1]) / out_height

     # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로   
    # 코 
    if prob1 > 0.1 :    
        points.append((int(x1), int(y1)))

    else:  # [not pointed]
        points.append(None)
        #print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    # 목
    if prob2 > 0.1 :
        points.append((int(x2), int(y2)))
    else:  # [not pointed]
        points.append(None)

    # 오른쪽 어깨
    if prob3 > 0.1 :    
        points.append((int(x3), int(y3)))
    else:  # [not pointed]
        points.append(None)
    
    # 왼쪽 어깨
    if prob4 > 0.1 :    
        points.append((int(x4), int(y4)))
    else:  # [not pointed]
        points.append(None)

    # 오른쪽 귀
    if prob5 > 0.1 :    
        points.append((int(x5), int(y5)))
    else:  # [not pointed]
        points.append(None)

    # 오른쪽 귀
    if prob6 > 0.1 :    
        points.append((int(x6), int(y6)))
    else:  # [not pointed]
        points.append(None)

    for (x,y,w,h) in faces:
        # opencv 이미지 > dlib용 사각형 변환
        dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        # 랜드마크 포인트 지정
        landmarks = np.matrix([[p.x,p.y] for p in predictor(frame, dlib_rect).parts()])
        # 원하는 포인트 넣음 (현재 전부)
        landmarks_display = landmarks[0:68]
        
        #양 턱 시작, 턱 끝, 코 끝
        points.append(landmarks[JAWLINE_POINTS[0]])
        points.append(landmarks[JAWLINE_POINTS[8]])
        points.append(landmarks[JAWLINE_POINTS[16]])
        points.append(landmarks[NOSE_POINTS[3]])

    return frame

def output_keypoints_with_lines(frame, POSE_PAIRS):
    global y_list
 
    try:
        # 어깨 중점
        cen_sholder_X = (points[2][0] + points[3][0]) // 2
        cen_sholder_Y = (points[2][1] + points[3][1]) // 2

        cen_sholder = (cen_sholder_X, cen_sholder_Y)
    
        cen_eyes_X = (points[4][0] + points[5][0]) // 2
        cen_eyes_Y = (points[4][1] + points[5][1]) // 2

        cen_eyes = (cen_eyes_X, cen_eyes_Y)

         # 광대 중점과 어깨 중점 연결
        if cen_eyes and cen_sholder:
            y_list.append(abs(cen_eyes_Y - cen_sholder_Y))
        
        if points[7] and cen_sholder:
            y_list.append(abs(points[7][1]-cen_sholder_Y))
            

    except (TypeError):
        cen_eyes_X = (points[4][0] + points[5][0]) // 2
        cen_eyes_Y = (points[4][1] + points[5][1]) // 2

        cen_eyes = (cen_eyes_X, cen_eyes_Y)

        if points[1] and cen_eyes:
            y_list.append(abs(cen_eyes_Y - points[1][1]))
            
        if points[7] and cen_sholder:
            y_list.append(abs(points[7][1]-cen_sholder_Y))

    
    return frame
# 네트워크 불러오기
net = cv.dnn.readNetFromCaffe(proto_file, weights_file)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
def output_keypoints_with_lines_video(proto_file, weights_file, BODY_PARTS, POSE_PAIRS, net):
    

    

    
    load_img2 = "./dataset/forw-samples/"
    save_img2 = "./dataset/forw_posture/forward-"
    for img in os.listdir(load_img2):
        
        aa = load_img2 + img
        cap = cv.imread(aa, cv.IMREAD_COLOR)

        cap = output_keypoints(frame=cap, net=net, BODY_PARTS=BODY_PARTS)
        cap = output_keypoints_with_lines(frame=cap, POSE_PAIRS=POSE_PAIRS)

    load_img1 = "./dataset/corr-samples/"
    save_img1 = "./dataset/corr_posture/correct-"

    for img in os.listdir(load_img1):
        
        aa = load_img1 + img
        cap = cv.imread(aa, cv.IMREAD_COLOR)

        cap = output_keypoints(frame=cap, net=net, BODY_PARTS=BODY_PARTS)
        cap = output_keypoints_with_lines(frame=cap, POSE_PAIRS=POSE_PAIRS)

# 키포인트를 저장할 빈 리스트
points = []
y_list = []

output_keypoints_with_lines_video(proto_file=protoFile_coco, weights_file=weightsFile_coco, BODY_PARTS=BODY_PARTS_COCO, POSE_PAIRS=POSE_PAIRS_COCO, net=net)

cv.waitKey(0) # esc 입력시 종료
cv.destroyAllWindows()
