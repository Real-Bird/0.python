import json
import cv2 as cv

img = cv.imread("./dataset/corr-samples/91.jpg", cv.IMREAD_COLOR)
with open("img1.json", "r") as st_json:

    st_python = json.load(st_json)

aa = st_python["people"][0]["pose_keypoints"]

cv.circle(img, (int(aa[0]), int(aa[1])), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
cv.putText(img, "{}".format(1), (int(aa[0]), int(aa[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv.LINE_AA)

cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()