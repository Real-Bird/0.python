from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model    #학습된 모델 로드

######################################################################
#이미지 전처리
#######################################################################

def stream_gen(frame):

    # 이미지를 4D 텐서로 변경
    img = image.load_img(frame, target_size =(150, 150))
    # print(img )
    img_tensor = image.img_to_array(img)
    # print(img_tensor.shape)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.  # 모델이 훈련될 때 입력에 적용한 전처리 방식을 동일하게 사용합니다
    # print(img_tensor.shape)  # 이미지 텐서의 크기는 (1, 150, 150, 3)입니다

#######################################################################
    model = load_model("tn_model.h5")

    result = model.predict( img_tensor ) 

    if result > 0.5 :
        print("거북목")
    else :
        print("정자세")