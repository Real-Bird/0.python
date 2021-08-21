import os
 
def changeName(path, cName):
    i = 0
    for filename in os.listdir(path):
        os.rename(path+filename, path+str(cName)+str(i)+'.jpg')
        i += 1

# changeName('./dataset/data_total/','_') 
changeName('./dataset/copy_corr/','c') 
changeName('./dataset/copy_forw/','f') 
# changeName('D:/jb_python/self_study/boyoung/','boyoung')
# changeName('D:/jb_python/self_study/gongyu/','gongyu')
# changeName('D:/jb_python/self_study/insung/','insung')
# changeName('D:/jb_python/self_study/iu/','iu')
# changeName('D:/jb_python/self_study/jongseock/','jongseock')
# changeName('D:/jb_python/self_study/mino/','mino')
# changeName('D:/jb_python/self_study/minyoung/','minyoung')
# changeName('D:/jb_python/self_study/nara/','nara')
# changeName('D:/jb_python/self_study/shinhye/','shinhye')
# changeName('D:/jb_python/self_study/zico/','zico')