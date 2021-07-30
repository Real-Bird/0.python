# fashion_pose.py : MPII를 사용한 신체부위 검출
import cv2

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
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# 이미지 읽어오기
image = cv2.imread("man.jpg")

# frame.shape = 불러온 이미지에서 height, width, color 받아옴
imageHeight, imageWidth, _ = image.shape
 
# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
 
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
minVal1, prob1, minLoc1, point1 = cv2.minMaxLoc(probMap1)
minVal2, prob2, minLoc2, point2 = cv2.minMaxLoc(probMap2)
# 원래 이미지에 맞게 점 위치 변경
x1 = (imageWidth * point1[0]) / W
y1 = (imageHeight * point1[1]) / H
x2 = (imageWidth * point2[0]) / W
y2 = (imageHeight * point2[1]) / H

# 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
if prob1 > 0.1 :    
    cv2.circle(image, (int(x1), int(y1)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    cv2.putText(image, "{}".format(2), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    points.append((int(x1), int(y1)))
else :
    points.append(None)
if prob2 > 0.1 :
    cv2.circle(image, (int(x2), int(y2)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
    cv2.putText(image, "{}".format(5), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    points.append((int(x2), int(y2)))
else :
    points.append(None)


cv2.imshow("Output-Keypoints",image)
cv2.waitKey(0)

# 이미지 복사
imageCopy = image

# 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
# for pair in POSE_PAIRS:
#     partA = pair[0]             # Head
#     partA = BODY_PARTS[partA]   # 0
#     partB = pair[1]             # Neck
#     partB = BODY_PARTS[partB]   # 1
    
#     #print(partA," 와 ", partB, " 연결\n")
#     if points[partA] and points[partB]:
#         cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)

#print(partA," 와 ", partB, " 연결\n")
if points[0] and points[1]:
    cv2.line(imageCopy, points[0], points[1], (0, 255, 0), 2)

cv2.imshow("Output-Keypoints",imageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()