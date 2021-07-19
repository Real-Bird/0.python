import json
fileName = 'config.json'
 
class Options :
    def __init__ (self )  :
        cate1 = json.loads(open(fileName, 'rb').read().decode('utf-8'))
         
        self.raw_data_dir  = cate1["raw_data_dir"] 
        self.clean_data_dir  = cate1["clean_data_dir"]
        self.num_epochs = cate1["num_epochs"]
        self.embd_size =cate1["embd_size"] 
        self.max_len=cate1["max_len"]
        self.model_o=cate1["model_o"]
        self.model_e=cate1["model_e"] 
        self.max_unique_word = cate1["max_unique_word"]