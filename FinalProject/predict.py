from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model    #학습된 모델 로드

######################################################################
#이미지 전처리
#######################################################################

def stream_gen(img, model_path):
    
    print( 1 )
    img_tensor = image.img_to_array(img)
    print( 2 )
    img_tensor = np.expand_dims(img_tensor, axis=0)
    print(3)
    img_tensor /= 255.  # 모델이 훈련될 때 입력에 적용한 전처리 방식을 동일하게 사용합니다
    # print(img_tensor.shape)  # 이미지 텐서의 크기는 (1, 150, 150, 3)입니다
    print(4)
#######################################################################
    model = load_model(model_path)
    print(5)
    result = model.predict( img_tensor ) 
    print(6)
    if result > 0.5 :
        print("forward")
    else :
        print("correct")

    return result