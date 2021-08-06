import os
 
def changeName(path, cName):
    i = 1
    for filename in os.listdir(path):
        os.rename(path+filename, path+str(cName)+str(i)+'.jpg')
        i += 1
 
changeName('./dataset/corr-samples/','correct_')