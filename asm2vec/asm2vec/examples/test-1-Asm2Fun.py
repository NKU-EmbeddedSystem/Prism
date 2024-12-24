import os
import numpy as np
import pandas as pd
import time
# extract function name
def preprocess_arm():
    function_name_list = []
    arm_path = "/home/cy/ccd/dataset/ASM/RISC"
    count = 0
    for root, dirs, files in os.walk(arm_path):
        dirs.sort(key=lambda x: int(x))
        for d in dirs:
            tmp_function_name_disk = {}
            new_path = os.path.join(arm_path, d)
            for root1, dirs1, files1 in os.walk(new_path):
                files1.sort(key=lambda x: int(x.split(".")[0]))
                for fi in files1:
                    os.chdir(new_path)
                    f = open(fi, "r")
                    lines = f.readlines()
                    flaga=0
                    tmp_list = []
                    tmp_function_name_list = []
                    for line in lines:
                        if "%function" in line:
                            la=flaga
                            tmp_list.append(la)
                        flaga=flaga+1
                    for i in tmp_list:
                        file=lines[int(i)]
                        str=file.split(",")[0]
                        function_name=str.split("\t")[-1]
                        tmp_function_name_list.append(function_name)
                        count=count+1
                    #print(function_name_list)
                    tmp_function_name_disk[fi]=tmp_function_name_list
                #print(tmp_function_name_disk)
                function_name_list.append(tmp_function_name_disk)
    print(len(function_name_list))
    print(function_name_list)
    print(count)
    # breakpoint()
    # os.makedirs('/home/cy/ccd/dataset/ASM/Risc_examples', exist_ok=True)
    # np.save('/home/cy/ccd/dataset/ASM/Risc_examples/function_list_arm_test.npy', function_name_list)
def load_arm():
    func_list = np.load('/home/cy/ccd/dataset/ASM/Risc_Function/armFunctionList.npy', allow_pickle=True)
    #print(func_list)
    count=0
    for i in func_list:
        for  j in i.values():
            #print(j)
            count=count+len(j)
    print("arm:"+str(count))
def preprocess_x86():
    function_name_list = []
    x86_path = "/home/cy/ccd/dataset/ASM/CISC"
    count = 0
    for root, dirs, files in os.walk(x86_path):
        if ".DS_Store" in dirs:
            dirs.remove(".DS_Store")
        dirs.sort(key=lambda x: int(x))
        for d in dirs:
            tmp_function_name_disk = {}
            new_path = os.path.join(x86_path, d)
            for root1, dirs1, files1 in os.walk(new_path):
                if ".DS_Store" in files1:
                    files1.remove(".DS_Store")
                files1.sort(key=lambda x: int(x.split(".")[0]))
                for fi in files1:
                    os.chdir(new_path)
                    f = open(fi, "r")
                    lines = f.readlines()
                    flaga=0
                    tmp_list = []
                    tmp_function_name_list = []
                    for line in lines:
                        if "@function" in line:
                            la=flaga
                            tmp_list.append(la)
                        flaga=flaga+1
                    for i in tmp_list:
                        file=lines[int(i)]
                        str=file.split(",")[0]
                        function_name=str.split("\t")[-1]
                        tmp_function_name_list.append(function_name)
                        count=count+1
                    #print(function_name_list)
                    tmp_function_name_disk[fi]=tmp_function_name_list
                #print(tmp_function_name_disk)
                function_name_list.append(tmp_function_name_disk)
    print(len(function_name_list))
    print(function_name_list)
    print(count)
    os.makedirs('/home/cy/ccd/dataset/ASM/Cisc_examples', exist_ok=True)
    np.save('/home/cy/ccd/dataset/ASM/Cisc_examples/function_list_x86_test.npy', function_name_list)
def load_x86():
    func_list = np.load('/home/cy/ccd/dataset/ASM/Cisc_Function/x86FunctionList.npy', allow_pickle=True)
    #print(func_list)
    count=0
    for i in func_list:
        for j in i.values():
            #print(j)
            count=count+len(j)
    print("x86:"+str(count))

if __name__=="__main__":
    starttime=time.time()
    # preprocess_arm()
    load_arm()
    # preprocess_x86()
    load_x86()
    endtime=time.time()
    print('extract function name time :'+str(endtime-starttime))