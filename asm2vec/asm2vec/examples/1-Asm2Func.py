import os
import time
import numpy as np
import pandas as pd

#extract function name from asm code
def ExtractArmFunctionName():
    armFunctionName = []
    armAsmCodePath = "/home/cy/ccd/dataset/GCJ-ASM/RISC_Fina"
    functionCount = 0
    for root, dirs, files in os.walk(armAsmCodePath):
        #root: /home/cy/ccd/dataset/ASM/RISC
        #dirs: unsorted list of dir names
        #files: null
        dirs.sort(key=lambda x: int(x))#sort dirs by name
        for dir in dirs:
            functionNameDict = {}
            filePath = os.path.join(armAsmCodePath, dir)
            for root1, dirs1, codeFile in os.walk(filePath):
                #codeFile: unsorted list of file names
                codeFile.sort(key=lambda x: int(x.split(".")[0]))
                for code in codeFile:
                    os.chdir(filePath)
                    f = open(code, "r")
                    lines = f.readlines()
                    tmpList = []
                    tmpFuncNameList = []
                    funcNameLineNum = 0
                    for line in lines:
                        #line: asm code line
                        if "%function" in line:
                            tmpList.append(funcNameLineNum)
                        funcNameLineNum = funcNameLineNum + 1
                    for i in tmpList:
                        asmLine = lines[int(i)]
                        str = asmLine.split(",")[0]
                        tmpFuncNameList.append(str.split("\t")[-1])
                        functionCount = functionCount + 1
                    functionNameDict[code] = tmpFuncNameList
                armFunctionName.append(functionNameDict)
    os.makedirs("/home/cy/ccd/dataset/GCJ-ASM/Risc_Fina_Function", exist_ok=True)
    np.save("/home/cy/ccd/dataset/GCJ-ASM/Risc_Fina_Function/armFunctionList.npy", armFunctionName)
    print(len(armFunctionName))
    # print(armFunctionName)
    print(functionCount)

def CountArmFunctionName():
    armFunctionName = np.load("/home/cy/ccd/dataset/GCJ-ASM/Risc_Fina_Function/armFunctionList.npy", allow_pickle=True)
    num = 0
    for i in armFunctionName:
        for j in i.values():
            num = num + len(j)
    print("arm:" + str(num))

def ExtractX86FunctionName():
    x86FunctionName = []
    x86AsmCodePath = "/home/cy/ccd/dataset/GCJ-ASM/CISC_Fina"
    functionCount = 0
    for root, dirs, files in os.walk(x86AsmCodePath):
        #root: /home/cy/ccd/dataset/ASM/CISC
        #dirs: unsorted list of dir names
        #files: null
        dirs.sort(key=lambda x: int(x))#sort dirs by name
        for dir in dirs:
            functionNameDict = {}
            filePath = os.path.join(x86AsmCodePath, dir)
            for root1, dirs1, codeFile in os.walk(filePath):
                #codeFile: unsorted list of file names
                codeFile.sort(key=lambda x: int(x.split(".")[0]))
                for code in codeFile:
                    os.chdir(filePath)
                    f = open(code, "r")
                    lines = f.readlines()
                    tmpList = []
                    tmpFuncNameList = []
                    funcNameLineNum = 0
                    for line in lines:
                        #line: asm code line
                        # print(line)
                        if "@function" in line:
                            tmpList.append(funcNameLineNum)
                        funcNameLineNum = funcNameLineNum + 1
                    for i in tmpList:
                        asmLine = lines[int(i)]
                        str = asmLine.split(",")[0]
                        tmpFuncNameList.append(str.split("\t")[-1])
                        functionCount = functionCount + 1
                    functionNameDict[code] = tmpFuncNameList
                x86FunctionName.append(functionNameDict)
    os.makedirs("/home/cy/ccd/dataset/GCJ-ASM/Cisc_Fina_Function", exist_ok=True)
    np.save("/home/cy/ccd/dataset/GCJ-ASM/Cisc_Fina_Function/x86FunctionList.npy", x86FunctionName)
    print(len(x86FunctionName))
    # print(x86FunctionName)
    print(functionCount)

def CountX86FunctionName():
    x86FunctionName = np.load("/home/cy/ccd/dataset/GCJ-ASM/Cisc_Fina_Function/x86FunctionList.npy", allow_pickle=True)
    num = 0
    for i in x86FunctionName:
        for j in i.values():
            num = num + len(j)
    print("x86:" + str(num))

if __name__ == "__main__":
    startTime = time.time()
    ExtractArmFunctionName()
    CountArmFunctionName()
    ExtractX86FunctionName()
    CountX86FunctionName()
    endTime = time.time()
    print("Extract Function Name time: %f s"%(endTime - startTime))