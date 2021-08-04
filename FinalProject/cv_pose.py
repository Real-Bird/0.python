import numpy as np
import cv2 as cv

protoFile_coco = "D:/Program Files/openpose-master/models/pose/coco/pose_deploy_linevec.prototxt"
weightsFile_coco = "D:/Program Files/openpose-master/models/pose/coco/pose_iter_440000.caffemodel"

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

def output_keypoints(frame, net, threshold, BODY_PARTS, now_frame, total_frame):
    global points
    
    # 입력 이미지의 사이즈 정의
    image_height = 640
    image_width = 480

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
    probMap5 = out[0, 16, :, :]
    probMap6 = out[0, 17, :, :]

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
        cv.circle(frame, (int(x1), int(y1)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(1), (int(x1), int(y1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x1), int(y1)))

    else:  # [not pointed]
        points.append(None)
        #print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    # 목
    if prob2 > 0.1 :
        cv.circle(frame, (int(x2), int(y2)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(2), (int(x2), int(y2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x2), int(y2)))
    else:  # [not pointed]
        points.append(None)

    # 오른쪽 어깨
    if prob3 > 0.1 :    
        cv.circle(frame, (int(x3), int(y3)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(3), (int(x3), int(y3)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x3), int(y3)))
    else:  # [not pointed]
        points.append(None)
    
    # 왼쪽 어깨
    if prob4 > 0.1 :    
        cv.circle(frame, (int(x4), int(y4)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(4), (int(x4), int(y4)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x4), int(y4)))
    else:  # [not pointed]
        points.append(None)

    # 오른쪽 귀
    if prob5 > 0.1 :    
        cv.circle(frame, (int(x5), int(y5)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(5), (int(x5), int(y5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x5), int(y5)))
    else:  # [not pointed]
        points.append(None)

    # 오른쪽 귀
    if prob6 > 0.1 :    
        cv.circle(frame, (int(x6), int(y6)), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv.putText(frame, "{}".format(6), (int(x6), int(y6)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)
        points.append((int(x6), int(y6)))
    else:  # [not pointed]
        points.append(None)

    return frame

def output_keypoints_with_lines(frame, POSE_PAIRS):
    
    if points[2] and points[3]:
        #print(f"[linked] {points[1]} <=> {points[3]}")
        cv.line(frame, points[2], points[3], (0, 255, 0), 3)  

    if points[4] and points[5]:
        cv.line(frame, points[4], points[5], (0, 255, 0), 3)  
    try:
        # 어깨 중점
        cen_sholder_X = (points[2][0] + points[3][0]) // 2
        cen_sholder_Y = (points[2][1] + points[3][1]) // 2

        cen_sholder = (cen_sholder_X, cen_sholder_Y)
    
        cen_ear_X = (points[4][0] + points[5][0]) // 2
        cen_ear_Y = (points[4][1] + points[5][1]) // 2

        cen_ear = (cen_ear_X, cen_ear_Y)

         # 광대 중점과 어깨 중점 연결
        if cen_ear and cen_sholder:
            cv.line(frame, cen_ear, cen_sholder, (0, 0, 255), 3)

    except (TypeError):
        try:
            cen_ear_X = (points[4][0] + points[5][0]) // 2
            cen_ear_Y = (points[4][1] + points[5][1]) // 2

            cen_ear = (cen_ear_X, cen_ear_Y)

            if points[1] and cen_ear:
                cv.line(frame, points[1], cen_ear, (0, 255, 0), 3)
        except:
            pass
    return frame

def output_keypoints_with_lines_video(proto_file, weights_file, threshold, BODY_PARTS, POSE_PAIRS):

    # 네트워크 불러오기
    net = cv.dnn.readNetFromCaffe(proto_file, weights_file)

    # openCV CUDA 없을 경우 주석 처리
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv.VideoCapture(0)
    
    while True:
        now_frame_boy = capture.get(cv.CAP_PROP_POS_FRAMES)
        total_frame_boy = capture.get(cv.CAP_PROP_FRAME_COUNT)
        
        if now_frame_boy == total_frame_boy:
            break

        ret, frame_boy = capture.read()
        
        frame_boy = cv.flip(frame_boy,1)
        frame_boy = output_keypoints(frame=frame_boy, net=net, threshold=threshold, BODY_PARTS=BODY_PARTS, now_frame=now_frame_boy, total_frame=total_frame_boy)
        frame_boy = output_keypoints_with_lines(frame=frame_boy, POSE_PAIRS=POSE_PAIRS)
        cv.imshow("Output_Keypoints", frame_boy)

        if cv.waitKey(1) == 27:  # esc 입력시 종료
            break

    capture.release()
    cv.destroyAllWindows()

# 키포인트를 저장할 빈 리스트
points = []

# output_keypoints_with_lines_video(proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
#                                   threshold=0.1, BODY_PARTS=BODY_PARTS_BODY_25, POSE_PAIRS=POSE_PAIRS_BODY_25)

output_keypoints_with_lines_video(proto_file=protoFile_coco, weights_file=weightsFile_coco,
                                 threshold=0.1, BODY_PARTS=BODY_PARTS_COCO, POSE_PAIRS=POSE_PAIRS_COCO)