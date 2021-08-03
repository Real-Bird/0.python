import os
import dlib
import glob
import cv2  #opencv 사용

def swapRGB2BGR(rgb):
    r, g, b = cv2.split(img)
    bgr = cv2.merge([b,g,r])
    return bgr

# 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
detector = dlib.get_frontal_face_detector()
# 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 이미지를 화면에 표시하기 위한 openCV 윈도 생성
cv2.namedWindow('Face')
# 파일에서 이미지 불러오기
img = dlib.load_rgb_image("photo.jpg")      

#불러온 이미지 데이터를 R과 B를 바꿔준다.
cvImg = swapRGB2BGR(img)    

cvImg = cv2.resize(cvImg, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

dets = detector(img, 1)

# 인식된 좌표에서 랜드마크 추출 
shape = predictor(img, dets[0])
#print(shape.num_parts)
# num_parts(랜드마크 구조체)를 하나씩 루프를 돌린다.
for i in range(0, shape.num_parts):
    # 해당 X,Y 좌표를 두배로 키워 좌표를 얻고
    x = shape.part(i).x*2
    y = shape.part(i).y*2

    # 좌표값 출력
    print(str(x) + " " + str(y))

    # 이미지 랜드마크 좌표 지점에 인덱스(랜드마크번호, 여기선 i)를 putText로 표시해준다.
    cv2.putText(cvImg, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))                    
# 랜드마크가 표시된 이미지를 openCV 윈도에 표시
cv2.imshow("Face",cvImg)
cv2.waitKey(0)
