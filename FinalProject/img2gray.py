import cv2 as cv
import os

load_img = "./dataset/new_train_samples/forward_posture/"
save_img = "./dataset/new_train_samples/forw_gray/forw_gray_"
for img in os.listdir(load_img):
    
    aa = load_img + img
    cap = cv.imread(aa, cv.IMREAD_COLOR)
    gray = cv.cvtColor(cap, cv.COLOR_BGR2GRAY)
    cv.imwrite(save_img+img, gray)

cv.waitKey(0) # esc 입력시 종료
cv.destroyAllWindows()