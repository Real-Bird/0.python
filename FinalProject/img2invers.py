import cv2 as cv
import os

img = cv.imread("photo.jpg")

out = img.copy()

out = 255 - out

cv.imshow("original", img)

cv.imshow("flip", out)

cv.waitKey(0)

load_img = "./dataset/corr-samples/"
save_img = "./dataset/corr_posture_inversion/correct_invers_"
for img in os.listdir(load_img):
    
    aa = load_img + img
    cap = cv.imread(aa, cv.IMREAD_COLOR)
    invers = 255 - cap
    cv.imwrite(save_img+img, invers)

cv.waitKey(0) # esc 입력시 종료
cv.destroyAllWindows()