from bert_models import BertModel, BertConfig
from tokenization import BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
state_dict = torch.load("/home/cy/ccd/Bert-Fine-Tune/Prism-fine-tune/finetune/model-state-dict/code-mrpc-weight-freeze.pt")
#state_dict=state_dict1["emb.weight"]
del state_dict['classifier.weight']
del state_dict['classifier.bias']
config = BertConfig(vocab_size_or_config_json_file=30522, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
# config = BertConfig.from_pretrained('/home/cy/ccd/Bert-Fine-Tune/Prism-fine-tune/finetune/MRPC/train_epoch_2000-weight-gcjtest')
# ——————构造模型——————
class TextNet(nn.Module):
    def __init__(self):
        super(TextNet, self).__init__()
        self.bert = BertModel(config)
        #self.dropout = nn.Dropout(p=0.1)
        self.emb=nn.Linear(self.bert.config.hidden_size,96)
        self.activation = nn.Tanh()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        emb = self.activation(self.emb(pooled_output))
        return emb

def get_bert_96(text):
    textNet = TextNet()
    #mode_disk=textNet.state_dict()
    #mode_disk.update(state_dict)
    #mode_disk=textNet.emb.state_dict()
    #print(mode_disk)
    #print(mode_disk)
    #textNet.emb.load_state_dict(mode_disk)
    #textNet.emb.load_state_dict(textNet.state_dict()['emb.weight'].__deepcopy__(state_dict1))
    #textNet.emb.load_state_dict(state_dict,strict=False)
    textNet.load_state_dict(state_dict,strict=False)
    tokenizer = BertTokenizer.from_pretrained('/home/cy/ccd/Bert-Fine-Tune/pretrained_models')
    #f=open(file,"r")
    #text=f.read()
    texts = "[CLS] " + text + " [SEP]"
    # 然后进行分词
    tokenized_text1 = tokenizer.tokenize(texts)
    indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
    # breakpoint()
    # 分词结束后获取BERT模型需要的tensor
    segments_ids1 = [0] * len(tokenized_text1)
    tokens_tensor1 = torch.tensor([indexed_tokens1])# 将list转为tensor
    segments_tensors1 = torch.tensor([segments_ids1])
    # 获取所有词向量的embedding
    #word_vectors1 = bertmodel(tokens_tensor1, segments_tensors1)[0]
    #print(word_vectors1.shape)
    # 获取句子的embedding
    sentenc_vector1 = textNet(tokens_tensor1, segments_tensors1)
    print(sentenc_vector1.shape)
    sentenc_vector1=sentenc_vector1[0].tolist()
    # print(sentenc_vector1)
    return str(sentenc_vector1)

def text_split(file):
    f4=open(file,"r")
    lines = f4.readlines()
    text1 = ""
    count= 0
    for line in lines:
        count=count+1
    print("count:"+str(count))
    if count >= 30:
        for i in range(30):
            text1 += lines[i]
    else:
        for i in range(15):
            text1 += lines[i]
    return text1

def text_split_1(file):
    f4=open(file,"r")
    lines = f4.readlines()
    text1 = ""
    count= 0
    for line in lines:
        count=count+1
    print("count:"+str(count))
    for i in range(10):
        text1 += lines[i]
    return text1

def Generate_feature_from_bert():
    bert_feature_path="/home/cy/ccd/dataset/GCJ_test_feature"
    lists=[]
    for root,dirs,files in os.walk("/home/cy/ccd/dataset/GCJ_test"):
        if not os.path.exists(bert_feature_path):
            os.mkdir(bert_feature_path)
        dirs.sort(key=lambda x:int(x))
        print(dirs)

        for d in dirs:
            folder_path=os.path.join(bert_feature_path,d)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            new_path=os.path.join(root,d)
            for root1,dirs1,files1 in os.walk(new_path):
                files1.sort(key=lambda x:int(x.split(".")[0]))
                for fi in files1:
                    fi_name = fi.strip().split(".")[0]
                    file_path=os.path.join(folder_path,fi_name)
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    file=new_path+"/"+fi
                    try:
                        f1=open(file,"r")
                        print(file)
                        text=f1.read()
                        text = text.replace("\n", "")
                        emb = get_bert_96(text)
                        # breakpoint()
                        print(file_path)
                        f2 = open(file_path + "/" + "0.txt", "w")
                        f2.write(emb)
                        f2.close()
                    except:
                        try:
                            print(file)
                            lists.append(file)
                            text_sp=text_split(file)
                            emb_sp = get_bert_96(text_sp)
                            f3 = open(file_path + "/" + "0.txt", "w")
                            f3.write(emb_sp)
                            f3.close()
                            print(file)
                            print("##############################################")
                        except:
                            text_sp1=text_split_1(file)
                            emb_sp = get_bert_96(text_sp1)
                            f3 = open(file_path + "/" + "0.txt", "w")
                            f3.write(emb_sp)
                            f3.close()
                            print(file)
                            print("##############################################")
    print('ERR_lists :',lists)


if __name__ == '__main__':
    starttime=time.time()
    # breakpoint()
    Generate_feature_from_bert()
    endtime=time.time()
    print("Generate embedding time:"+str(endtime-starttime))

