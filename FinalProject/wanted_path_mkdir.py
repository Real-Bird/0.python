# 원하는 위치에 같은 이름의 디렉토리 만드는 코드

import os

name = input("What the pose : ")
directory = name
parent_dir = "D:/jb_python/FinalProject/dataset/face_train/"
path = os.path.join(parent_dir, directory)
os.mkdir(path)
parent_dir = "D:/jb_python/FinalProject/dataset/face_test/"
path = os.path.join(parent_dir, directory)
os.mkdir(path)