import numpy as np
import os
import asm2vec.asm
import asm2vec.parse
import asm2vec.model
import pandas as pd
import time
import sys
sys.setrecursionlimit(5000)

def train_arm():
    armFunctionNamePath = "/home/cy/ccd/dataset/GCJ-ASM/Risc_Fina_Function/armFunctionList.npy"
    armFunctionNameList = np.load(armFunctionNamePath, allow_pickle=True)
    # print(armFunctionNameList[0])
    trainingFunctions = []
    functionNums = []
    armCodePath = "/home/cy/ccd/dataset/GCJ-ASM/RISC_Fina/"
    for root, dirs, files in os.walk(armCodePath):
        #root: /home/cy/ccd/dataset/ASM/RISC
        #dirs: unsorted list of dir names
        #files: null
        dirs.sort(key=lambda x: int(x.split('.')[0]))
        count = 0
        for d in dirs: 
            tmpFunctionDict = {}
            filePath = os.path.join(root, d)
            for root1, dirs1, codeFile in os.walk(filePath):
                codeFile.sort(key=lambda x: int(x.split('.')[0]))
                for file in codeFile:  # 遍历d中的文件
                    #tmpFunctionDict: {'xxx/xx.s: [func1, func2]}
                    tmpFunctionDict[os.path.join(filePath, file)] = len(armFunctionNameList[count][file])
                    # breakpoint()
                    tmpFunctions = asm2vec.parse.parse(os.path.join(filePath, file),func_names=armFunctionNameList[count][file])
                    trainingFunctions.extend(tmpFunctions)
            functionNums.append(tmpFunctionDict)
            count = count + 1
            # print(count)
    # print(functionNums[0])
    # functionNums: [{path/to/xxx.s : num}]
    np.save('/home/cy/ccd/dataset/GCJ-ASM/Risc_Fina_Function/armFunctionPath2Num.npy', functionNums)

    print('Number of training functions:', len(trainingFunctions))
    model = asm2vec.model.Asm2Vec(d=48)
    trainingRepo = model.make_function_repo(trainingFunctions)
    model.train(trainingRepo)
    print('Asm2Vec Model has been Training completely.')

    for tf in trainingRepo.funcs():
        
        # print('Norm of trained function "{}" = {}'.format(tf.sequential().name(), np.linalg.norm(tf.v)))
        # break
        df = pd.DataFrame(tf.v).T
        df.to_csv("/home/cy/ccd/dataset/GCJ-ASM/Risc_Fina_Function/armFunctionVec.csv", encoding="utf-8", mode="a", header=False, index=False)
    print("Arm finish!")

def train_x86():
    x86FunctionNamePath = "/home/cy/ccd/dataset/GCJ-ASM/Cisc_Fina_Function/x86FunctionList.npy"
    x86FunctionNameList = np.load(x86FunctionNamePath, allow_pickle=True)
    # print(armFunctionNameList[0])
    trainingFunctions = []
    functionNums = []
    x86CodePath = "/home/cy/ccd/dataset/GCJ-ASM/CISC_Fina/"
    for root, dirs, files in os.walk(x86CodePath):
        #root: /home/cy/ccd/dataset/ASM/CISC
        #dirs: unsorted list of dir names
        #files: null
        dirs.sort(key=lambda x: int(x.split('.')[0]))
        count = 0
        for d in dirs: 
            tmpFunctionDict = {}
            filePath = os.path.join(root, d)
            for root1, dirs1, codeFile in os.walk(filePath):
                codeFile.sort(key=lambda x: int(x.split('.')[0]))
                for file in codeFile:  # 遍历d中的文件
                    #tmpFunctionDict: {'xxx/xx.s: [func1, func2]}
                    tmpFunctionDict[os.path.join(filePath, file)] = len(x86FunctionNameList[count][file])
                    tmpFunctions = asm2vec.parse.parse(os.path.join(filePath, file),func_names=x86FunctionNameList[count][file])
                    trainingFunctions.extend(tmpFunctions)
            functionNums.append(tmpFunctionDict)
            count = count + 1
            # if(count == 4):
            #     breakpoint()
    # print(functionNums[0])
    # functionNums: [{path/to/xxx.s : num}]
    np.save('/home/cy/ccd/dataset/GCJ-ASM/Cisc_Fina_Function/x86FunctionPath2Num.npy', functionNums)

    print('Number of training functions:', len(trainingFunctions))
    model = asm2vec.model.Asm2Vec(d=48)
    trainingRepo = model.make_function_repo(trainingFunctions)
    model.train(trainingRepo)
    print('Asm2Vec Model has been Training completely.')

    for tf in trainingRepo.funcs():
        
        # print('Norm of trained function "{}" = {}'.format(tf.sequential().name(), np.linalg.norm(tf.v)))
        # break
        df = pd.DataFrame(tf.v).T
        df.to_csv("/home/cy/ccd/dataset/GCJ-ASM/Cisc_Fina_Function/x86FunctionVec.csv", encoding="utf-8", mode="a", header=False, index=False)
    print("x86 finish!")

if __name__ == '__main__':
    starttime=time.time()
    train_arm()
    train_x86()
    endtime=time.time()
    runtime=endtime-starttime
    print("train asm2Vec time : "+str(runtime))
