#video_server.py
from simple_websocket_server import WebSocketServer, WebSocket
import base64, cv2
import numpy as np
import warnings
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model    #학습된 모델 로드
from predict import stream_gen

warnings.simplefilter("ignore", DeprecationWarning)

class SimpleEcho(WebSocket):
    
    def handle(self):
        print("a")
        
        msg = self.data
        print("b")
        img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        print("c")
        img = cv2.flip(img, 1)
        print("d")

        stream_gen(img)
        print("e")
        cv2.imshow("img", img)
        print("f")
        cv2.waitKey(1)

    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')

server = WebSocketServer('localhost', 3000, SimpleEcho)
server.serve_forever()