import os
 
def changeName(path, cName):
    i = 0
    for filename in os.listdir(path):
        os.rename(path+filename, path+str(cName)+str(i)+'.jpg')
        i += 1
 
changeName('./dataset/corr_test/','0')
changeName('./dataset/forw_test/','0')