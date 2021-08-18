import numpy as np
import cv2 as cv
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # mobilenet_v2 모델에 필요한 형식에 이미지를 적절하게 맞추기위한 함수(전처리)
from tensorflow.keras.preprocessing.image import img_to_array # 이미지를 numpy 배열로 변환
from tensorflow.keras.models import load_model # 모델 로드

t_vgg_pb = "D:\\jb_python\\FinalProject\\zz\\t_vgg_model_d1024.pb"

def t_predict(frame, net):
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

    percentage = float(out[0, 0]) * 100
    # bounding box 레이블 설정
    label = "co: " if out[0, 0] < 0.5 else "fo: "
    
    cv.rectangle(frame, (0, 0), (120, 40), (255, 255, 255), -1)
    # bounding box 출력
    if out[0, 0]  < 0.5:  
        cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, lineType=cv.LINE_AA)
        cv.putText(frame, "{:0.1f}".format(percentage), (50, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, lineType=cv.LINE_AA)
    else:
        cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv.LINE_AA)
        cv.putText(frame, "{:0.1f}".format(percentage), (50, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv.LINE_AA)
    

    return frame

def output_video(pb,threshold):

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

        frame = t_predict(frame, net)

        frame = imutils.resize(frame, width=640, height=480)

        cv.imshow("Output_Video", frame)
        
        

        if cv.waitKey(1) == 27:  # esc 입력시 종료
            break

    capture.release()
    cv.destroyAllWindows()

output_video(pb=t_vgg_pb, threshold=0.1)