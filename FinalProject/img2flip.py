import cv2 as cv
import os

load_img = "./dataset/tm_total/"
save_img = "./dataset/tm_total/flip_"
for img in os.listdir(load_img):
    
    aa = load_img + img
    cap = cv.imread(aa, cv.IMREAD_COLOR)
    flip = cv.flip(cap, 1)
    cv.imwrite(save_img+img, flip)

cv.waitKey(0) # esc 입력시 종료
cv.destroyAllWindows()