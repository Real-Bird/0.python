import cv2

def output_keypoints(frame, net, threshold, BODY_PARTS, now_frame, total_frame):
    global points

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

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

    #print(f"============================== frame: {now_frame:.0f} / {total_frame:.0f} ==============================")
    #for i in range(len(BODY_PARTS)):

    # 해당 신체부위 신뢰도 얻음.
    probMap1 = out[0, 0, :, :]
    probMap2 = out[0, 1, :, :]
    probMap3 = out[0, 2, :, :]
    probMap4 = out[0, 5, :, :]

    # global 최대값 찾기
    minVal1, prob1, minLoc1, point1 = cv2.minMaxLoc(probMap1)
    minVal2, prob2, minLoc2, point2 = cv2.minMaxLoc(probMap2)
    minVal3, prob3, minLoc3, point3 = cv2.minMaxLoc(probMap3)
    minVal4, prob4, minLoc4, point4 = cv2.minMaxLoc(probMap4)
    # 원래 이미지에 맞게 점 위치 변경
    x1 = (frame_width * point1[0]) / out_width
    y1 = (frame_height * point1[1]) / out_height
    x2 = (frame_width * point2[0]) / out_width
    y2 = (frame_height * point2[1]) / out_height
    x3 = (frame_width * point3[0]) / out_width
    y3 = (frame_height * point3[1]) / out_height
    x4 = (frame_width * point4[0]) / out_width
    y4 = (frame_height * point4[1]) / out_height

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로   
    # 코 
    if prob1 > 0.1 :    
        cv2.circle(frame, (int(x1), int(y1)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(frame, "{}".format(2), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x1), int(y1)))

    else:  # [not pointed]
        points.append(None)
        #print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    # 목
    if prob2 > 0.1 :
        cv2.circle(frame, (int(x2), int(y2)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(frame, "{}".format(5), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x2), int(y2)))
    else:  # [not pointed]
        points.append(None)

    # 오른쪽 어깨
    if prob3 > 0.1 :    
        cv2.circle(frame, (int(x3), int(y3)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(frame, "{}".format(2), (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x3), int(y3)))
    else:  # [not pointed]
        points.append(None)
    
    # 왼쪽 어깨
    if prob4 > 0.1 :    
        cv2.circle(frame, (int(x4), int(y4)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(frame, "{}".format(2), (int(x4), int(y4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x4), int(y4)))
    else:  # [not pointed]
        points.append(None)
    return frame

def output_keypoints_with_lines(frame, POSE_PAIRS):
       
    # 코와 목 접선
    if points[0] and points[1]:
        #print(f"[linked] {points[0]} <=> {points[1]}")
        cv2.line(frame, points[0], points[1], (0, 255, 0), 3)
    #else:
        #print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

    # 목과 오른쪽 어깨 접선
    if points[1] and points[2]:
        #print(f"[linked] {points[1]} <=> {points[2]}")
        cv2.line(frame, points[1], points[2], (0, 255, 0), 3)

    # 목과 왼쪽 어깨 접선
    if points[1] and points[3]:
        #print(f"[linked] {points[1]} <=> {points[3]}")
        cv2.line(frame, points[1], points[3], (0, 255, 0), 3)

    # print(points[1][1] - points[0][1])
    return frame

def output_keypoints_with_lines_video(proto_file, weights_file, threshold, BODY_PARTS, POSE_PAIRS):

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # openCV CUDA 없을 경우 주석 처리
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv2.VideoCapture(0)

    while True:
        now_frame_boy = capture.get(cv2.CAP_PROP_POS_FRAMES)
        total_frame_boy = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        if now_frame_boy == total_frame_boy:
            break

        ret, frame_boy = capture.read()
        frame_boy = cv2.flip(frame_boy,1)
        frame_boy = output_keypoints(frame=frame_boy, net=net, threshold=threshold, BODY_PARTS=BODY_PARTS, now_frame=now_frame_boy, total_frame=total_frame_boy)
        frame_boy = output_keypoints_with_lines(frame=frame_boy, POSE_PAIRS=POSE_PAIRS)
        cv2.imshow("Output_Keypoints", frame_boy)

        if cv2.waitKey(10) == 27:  # esc 입력시 종료
            break

    capture.release()
    cv2.destroyAllWindows()

BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]

# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_body_25 = "D:/Program Files/openpose-master/models/pose/body_25/pose_deploy.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_body_25 = "D:/Program Files/openpose-master/models/pose/body_25/pose_iter_584000.caffemodel"

# 키포인트를 저장할 빈 리스트
points = []

output_keypoints_with_lines_video(proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
                                  threshold=0.1, BODY_PARTS=BODY_PARTS_BODY_25, POSE_PAIRS=POSE_PAIRS_BODY_25)