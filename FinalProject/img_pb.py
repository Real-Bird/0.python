import numpy as np
import cv2 as cv
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # mobilenet_v2 모델에 필요한 형식에 이미지를 적절하게 맞추기위한 함수(전처리)
from tensorflow.keras.preprocessing.image import img_to_array # 이미지를 numpy 배열로 변환
from tensorflow.keras.models import load_model # 모델 로드


# 신체 path
t_vgg_pbtxt = "D:\\jb_python\\FinalProject\\zz\\t_vgg_model_d1024.pbtxt"
t_vgg_pb = "D:\\jb_python\\FinalProject\\zz\\t_vgg_model_d1024.pb"

# t_vgg_model = load_model("D:\\jb_python\\FinalProject\\tn_model\\t_vgg_model_d1024.h5")

def t_predict(frame, net):
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # RGB 변환

    # 프레임 resize
    frame = imutils.resize(frame, width=224, height=224)

    # frame = img_to_array(frame) # 이미지(얼굴)를 numpy 배열로 변환

    # frame = preprocess_input(frame) # 모델에 필요한 형식에 이미지(얼굴)를 적절하게 맞추기위한 함수(전처리)

    # print(frame.shape)

    # 입력 이미지의 사이즈 정의
    image_height = 224
    image_width = 224

    # 네트워크에 넣기 위한 전처리
    input_blob = cv.dnn.blobFromImage(frame, 1.0, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward() # shape : (1, 1)

    percentage = out[0, 0].astype("float32") * 100
    # bounding box 레이블 설정
    label = "co : " if out[0, 0] < 0.5 else "fo : "
    
    # bounding box 출력
    cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, lineType=cv.LINE_AA)
    cv.putText(frame, str(percentage), (50, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, lineType=cv.LINE_AA)
    cv.rectangle(frame, (0, 0), (200, 60), (255, 0, 0), 2)

    return frame

def output_video(pb,threshold):

    # 네트워크 불러오기
    net = cv.dnn.readNetFromTensorflow(pb)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    
    frame = cv.imread("D:/jb_python/FinalProject/dataset/forw-samples/322.jpg", cv.IMREAD_COLOR)

    frame = t_predict(frame, net)

    cv.imshow("Output_Video", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

output_video(pb=t_vgg_pb, threshold=0.1)