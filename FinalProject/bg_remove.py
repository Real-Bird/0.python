import cv2
from PIL import Image
import numpy as np
class UnsupportedFormat(Exception):
    def __init__(self, input_type):
        self.t = input_type
        def __str__(self):
    return " '{}'모드의 변환을 지원하지 않고 이미지 주소 (path), PIL.Image(pil) 모드 '.format(self.t)' 모드 ".format(self.t)
class MatteMatting():
    def __init__(self, original_graph, mask_graph, input_type='path'):
        """
        입력 이미지를 마스크를 통해 투명 맵 생성자로 변환
        :param original_graph: 이미지 주소 PIL 형식 CV2 형식을 입력하십시오.
        :param mask_graph: 그림 주소, PIL 형식 CV2 형식
        :param input_type: 입력 유형, path : 이미지 주소 pil : pil 유형, cv2 유형
        """
        if input_type == 'path':
        self.img1 = cv2.imread(original_graph)
        self.img2 = cv2.imread(mask_graph)
        elif input_type == 'pil':
        self.img1 = self.__image_to_opencv(original_graph)
        self.img2 = self.__image_to_opencv(mask_graph)
        elif input_type == 'cv2':
        self.img1 = original_graph
        self.img2 = mask_graph
        else:
        raise UnsupportedFormat(input_type)
    @staticmethod
    def __transparent_back(img):
        """
        :param img: 들어오는 그림 주소
        :return: 투명 다이어그램의 대체를 반환합니다
        """
        img = img.convert('RGBA')
        L, H = img.size
        color_0 = (255, 255, 255, 255) # 색상을 교체합니다
        for h in range(H):
        for l in range(L):
        dot = (l, h)
        color_1 = img.getpixel(dot)
        if color_1 == color_0:
            color_1 = color_1[:-1] + (0,)
            img.putpixel(dot, color_1)
        return img
    def save_image(self, path, mask_flip=False):
        """
        투명한 다이어그램을 저장하는 데 사용됩니다
        :param path: 위치 저장
        :param mask_flip: 마스크 펄터, 마스크의 흑백 색을 뒤집습니다.True 뒤집기;False 플립을 사용하지 마십시오
        """
        if mask_flip:
        img2 = cv2.bitwise_not(self.img2) # 흑백 뒤집기
        image = cv2.add(self.img1, img2)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # OpenCV PIL.Image 형식으로 변환합니다
        img = self.__transparent_back(image)
        img.save(path)
        @staticmethod
        def __image_to_opencv(image):
        """
        PIL.Image OpenCV 형식으로 변환합니다
        """
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return img

mm = MatteMatting("images.jpg", "mask.jpg")
mm.save_image("output.png", mask_flip=True) # mask_flip는 뒤집을 멜로디, 즉 검은 색으로 흰색을 돌리고 백색으로 변하고 있습니다.