import urllib.request
import os
import zipfile
import pandas as pd
import numpy as np
import pickle
'''下载模型 '''

model_dir = 'models/'
bert_model = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin'
bert_config = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json'
bert_vocab = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
    
def download_models():
    print("Downloading bert-base-uncased")
    urllib.request.urlretrieve(bert_model, model_dir+"pytorch_model.bin")
    print("Saved as bert-base-uncased-pytorch_model.bin")

    print("Downloading config for bert-base-uncased")
    urllib.request.urlretrieve(bert_config, model_dir+"bert_config.json")
    print("Saved config")

    print("Downloading uncased vocab")
    urllib.request.urlretrieve(bert_vocab, model_dir+'vocab.txt')
    print("Saved vocab")
    print("\tCompleted!")
 
download_models()
