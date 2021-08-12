#video_server.py
from simple_websocket_server import WebSocketServer, WebSocket
import base64, cv2
import numpy as np
import warnings
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model    #학습된 모델 로드

warnings.simplefilter("ignore", DeprecationWarning)
model = load_model("D:/jb_python/FinalProject/tn_model.h5")
class SimpleEcho(WebSocket):
    
    def handle(self):
        
        msg = self.data
        img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.flip(img, 1)

        img = image.load_img(img, target_size =(150, 150))

        img_tensor = image.img_to_array(img)

        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.  # 모델이 훈련될 때 입력에 적용한 전처리 방식을 동일하게 사용합니다

        result = model.predict( img_tensor ) 

        if result > 0.5 :
            print("거북목")
        else :
            print("정자세")
        cv2.imshow("img", img)
        cv2.waitKey(1)

        
        
    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')


server = WebSocketServer('localhost', 3000, SimpleEcho)
server.serve_forever()