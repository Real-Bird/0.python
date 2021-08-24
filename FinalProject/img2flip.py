import cv2 as cv
import os

def flip(load_img, save_img):

    for img in os.listdir(load_img):
    
        aa = load_img + img
        cap = cv.imread(aa, cv.IMREAD_COLOR)
        flip = cv.flip(cap, 1)
        cv.imwrite(save_img+img, flip)

cv.waitKey(0) # esc 입력시 종료
cv.destroyAllWindows()

co_img = "./dataset/copy_corr/"
co_save = "./dataset/copy_corr/flip_"
flip(co_img, co_save)

fo_img = "./dataset/copy_forw/"
fo_save = "./dataset/copy_forw/flip_"

flip(fo_img, fo_save)