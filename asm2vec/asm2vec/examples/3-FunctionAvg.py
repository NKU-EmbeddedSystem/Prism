import os
import numpy as np
import pandas as pd
import time



def find_path_arm():
    asm_path="/home/cy/ccd/dataset/GCJ-ASM/Fina_arm_syntax_feature"
    func_list = np.load('/home/cy/ccd/dataset/GCJ-ASM/Risc_Fina_Function/armFunctionList.npy', allow_pickle=True)
    df=pd.read_csv("/home/cy/ccd/dataset/GCJ-ASM/Risc_Fina_Function/armFunctionVec.csv",header=None)
    tmp_num_dict={}
    count=0

    for root,dirs,files in os.walk("/home/cy/ccd/dataset/GCJ-ASM/RISC_Fina"):
        if not os.path.exists(asm_path):
            os.mkdir(asm_path)
        dirs.sort(key=lambda x:int(x))
        print(dirs)
        # breakpoint()
        num=0
        for d in dirs:
            folder_path = os.path.join(asm_path,d)
            #print(folder_path)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            new_path = os.path.join(root,d)
            for root1,dirs1,files1 in os.walk(new_path):
                files1.sort(key=lambda x:int(x.split(".")[0]))
                for fi in files1:
                    fi_name = fi.strip().split(".")[0]
                    #print(fi_name)
                    asm_file_path = os.path.join(folder_path, fi_name)
                    if not os.path.exists(asm_file_path):
                        os.mkdir(asm_file_path)
                    row = len(func_list[num][fi])
                    tmp_num_dict[os.path.join(new_path, fi)] = row
                    item = df.iloc[count : count + row]
                    average = item.sum() / row
                    #print(average.tolist())
                    with open(asm_file_path + "/" + "0.txt", "w") as f:
                        f.write(str(average.tolist()))
                    # f.close()
                    count += row
            num=num+1
    #print(tmp_num_dict)
    #print(count)
    print("finish!")
    
def find_path_x86():
    asm_path="/home/cy/ccd/dataset/GCJ-ASM/Fina_x86_syntax_feature"
    func_list = np.load('/home/cy/ccd/dataset/GCJ-ASM/Cisc_Fina_Function/x86FunctionList.npy', allow_pickle=True)
    df=pd.read_csv("/home/cy/ccd/dataset/GCJ-ASM/Cisc_Fina_Function/x86FunctionVec.csv",header=None)
    tmp_num_dict={}
    count=0

    for root,dirs,files in os.walk("/home/cy/ccd/dataset/GCJ-ASM/CISC_fina"):
        if not os.path.exists(asm_path):
            os.mkdir(asm_path)
        dirs.sort(key=lambda x:int(x))
        #print(dirs)
        num=0
        for d in dirs:
            folder_path=os.path.join(asm_path,d)
            #print(folder_path)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            new_path=os.path.join(root,d)
            for root1,dirs1,files1 in os.walk(new_path):
                files1.sort(key=lambda x:int(x.split(".")[0]))
                for fi in files1:
                    fi_name=fi.strip().split(".")[0]
                    #print(fi_name)
                    asm_file_path=os.path.join(folder_path,fi_name)
                    if not os.path.exists(asm_file_path):
                        os.mkdir(asm_file_path)
                    row=len(func_list[num][fi])
                    tmp_num_dict[os.path.join(new_path,fi)]=row
                    item=df.iloc[count:count+row]
                    average=item.sum()/row
                    #print(average.tolist())
                    with open(asm_file_path+"/"+"0.txt","w") as f:
                        f.write(str(average.tolist()))
                    f.close()
                    count+=row
            num=num+1
    #print(tmp_num_dict)
    #print(count)
    print("finish!")


if __name__ == '__main__':
    starttime=time.time()
    # breakpoint()
    find_path_arm()
    find_path_x86()
    endtime=time.time()
    runtime=endtime-starttime
    print("function average time : "+str(runtime))