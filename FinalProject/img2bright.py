import numpy as np
import cv2
import os

load_img = "./dataset/tm_total/"
add_save_img = "./dataset/add/add_"
sub_save_img = "./dataset/sub/sub_"

for img in os.listdir(load_img):
    
    aa = load_img + img
    cap = cv2.imread(aa, cv2.IMREAD_COLOR)

    # 밝게
    # add_val = 70
    # add_array = np.full(cap.shape, (add_val, add_val, add_val), dtype=np.uint8)
    # add = cv2.add(cap, add_array)
    # cv2.imwrite(add_save_img+img, add)

    # 어둡게
    sub_val = 30
    sub_array = np.full(cap.shape, (sub_val, sub_val, sub_val), dtype=np.uint8)
    sub = cv2.subtract(cap, sub_array)
    
    cv2.imwrite(sub_save_img+img, sub)

cv2.waitKey(0) 
cv2.destroyAllWindows()