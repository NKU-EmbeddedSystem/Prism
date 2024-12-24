import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import sys
from model_loop import Classifier_Net
# from model_concat import Classifier_Net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
feature_len = 96
model_path="model/"
record_path=model_path+"5folder_fusion.txt"

# model
class Net(nn.Module):#设置网络
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output):
        super(Net, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)#五个hidden state
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.hidden5 = torch.nn.Linear(n_hidden4, n_hidden5)
        self.out = torch.nn.Linear(n_hidden5, n_output)


    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = self.out(x)
        return x

def main(train_path):
    start_time=time.time()
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    precision_list=[]
    recall_list=[]
    f1_list=[]
    accuracy_list=[]
    data_list=[]
    label_list=[]
    print("################### Load Data ####################")
    # Load Data
    # csv_data = pd.read_csv('../data/ARM_X86_syntax_semantic.csv',
    #                        sep=',', header=None)
    csv_data = pd.read_csv(train_path, sep=',', header=None)
    # breakpoint()
    # csv_data = pd.read_csv('/home/uuntu/ccd_dataset/meeting/bert_asm_O0_x86_O0/training_new/ARM_X86_syntax_semantic.csv', sep=',', header=None)
    data = csv_data.values.astype(np.float32)[0:200000, :-1]
    label = csv_data.values.astype(np.int32)[0:200000, -1:]
    total_data = np.hstack((data, label))
    np.random.shuffle(total_data)
    # print(total_data)
    total_size = len(total_data)
    part_size = int(total_size / 5)
    for i in range(0,5):
        data_list.append(total_data[i * part_size:(i + 1) * part_size, :-1])
        label_list.append(total_data[i * part_size:(i + 1) * part_size, -1])
    record = open(record_path, 'a')  # 以追加的方式记录结果

    print("#######Train Data###########")
    for num in range(5):
        train_start_time = time.time()
        net = Classifier_Net(feature_len=96, kernel_size = 3, n_output=2)
        # net = Net(n_feature=feature_len, n_hidden1=feature_len, n_hidden2=feature_len * 2, n_hidden3=feature_len * 4,
        #       n_hidden4=feature_len * 2, n_hidden5=feature_len,
        #       n_output=2)#这里传入的feature_len为64
        print(net)
        net.cuda()#將模型加载到GPU上去
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#学习率为0.001,构建好神经网络后，网络的参数都保存在parameters()函数当中
        loss_func = torch.nn.CrossEntropyLoss()#使用交叉熵损失函数

        if num == 0:
            tmp_data=data_list[1]
            tmp_label=label_list[1]

        else:
            tmp_data=data_list[0]
            tmp_label=label_list[0]
            #train_dataset = TensorDataset(tmp_data, tmp_label)
            #train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, num_workers=2)

        for i in range(5):
            if i == num:
                continue
            if num == 0 and i == 1:
                continue
            tmp_data = np.vstack((tmp_data, data_list[i]))
            tmp_label = np.hstack((tmp_label, label_list[i]))

        train_data = torch.from_numpy(tmp_data).float()
        train_label = torch.from_numpy(tmp_label).long()
        train_dataset = TensorDataset(train_data, train_label)
        train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, num_workers=2)

        test_datas = torch.from_numpy(data_list[num]).float()
        test_labels = torch.from_numpy(label_list[num]).long()


        print("#######Train Data###########")
        for epoch in range(80):
            print("epoch:", epoch)
            for i, data in enumerate(train_loader):
                train_data, train_label = data
                # print("epoch:",epoch,"item:",i,"train_data:",train_data.size(),"train_label:",train_label.size())
                out = net(train_data.cuda())
                loss = loss_func(out, train_label.cuda())
                # 每一遍epochs的过程都要一次用到这三个函数
                optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
                loss.backward()  # 反向传播计算得到每个参数的梯度值
                optimizer.step()  # 通过梯度下降执行一步参数更新
            print("loss:", loss.item())
            if epoch % 1 == 0:  # 每200轮计算一次
                prediction = torch.max(out, 1)[
                    1].cuda()  # out是softmax的输出，max()函数对softmax函数的输出值进行操作，求出预测值索引，然后与标签进行对比，计算准确率。
                pred_y = prediction.data.cpu().numpy()
                target_y = train_label.data.cpu().numpy()
                accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)  # 计算准确率

                tp = 0  # 预测为正例，实际为正例
                tn = 0  # 预测为负例，实际为负例
                fn = 0  # 预测为负例，实际为正例
                fp = 0  # 预测为正例，实际为负例
                for pred_id in range(len(pred_y)):
                    pred_cur = pred_y[pred_id]
                    target_cur = target_y[pred_id]
                    if (1 == pred_cur) and (1 == target_cur):
                        tp += 1
                    if (0 == pred_cur) and (0 == target_cur):
                        tn += 1
                    if (0 == pred_cur) and (1 == target_cur):
                        fn += 1
                    if (1 == pred_cur) and (0 == target_cur):
                        fp += 1
                if tp + fp != 0 and tp + fn != 0:
                    precision = tp / float(tp + fp)
                    recall = tp / float(tp + fn)
                    F1 = 2 * precision * recall / (precision + recall)
                    print("time=" + str(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())) + " precision=",
                          precision,
                          "recall=", recall, "F1=", F1, "accuracy=", accuracy)  # 输出评价指标

        torch.save(net.state_dict(), model_path + 'checkpoint_folder.pth')
        train_end_time = time.time()
        print("training time:" + str(train_end_time - train_start_time))

        test_start_time = time.time()
        out = net(test_datas.cuda())
        prediction = torch.max(out, 1)[1].cuda()
        pred_y = prediction.data.cpu().numpy()
        target_y = test_labels.data.cpu().numpy()

        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for pred_id in range(len(pred_y)):
            pred_cur = pred_y[pred_id]
            target_cur = target_y[pred_id]
            if (1 == pred_cur) and (1 == target_cur):
                tp += 1
            if (0 == pred_cur) and (0 == target_cur):
                tn += 1
            if (0 == pred_cur) and (1 == target_cur):
                fn += 1
            if (1 == pred_cur) and (0 == target_cur):
                fp += 1

        if tp + fp != 0 and tp + fn != 0:
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            F1 = 2 * precision * recall / (precision + recall)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(F1)
            accuracy_list.append(accuracy)
            print("time=" + str(
                time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())) + "\n" + "test_o0 result: " + "precision=",
                  precision,
                  "recall=", recall, "F1=", F1, "accuracy=", accuracy)
            record.write("time=" + str(
                time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())) + "\n" + "test_o0 result: " + "precision=" + str(
                precision) + " recall=" + str(recall) + " F1=" + str(F1) + " accuracy=" + str(accuracy) + "\n")
        test_end_time = time.time()
        print("prediction time:" + str(test_end_time - test_start_time))

    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    sum_accuracy = 0
    #loop num
    for i in range(5):
        sum_precision+=precision_list[i]
        sum_recall+=recall_list[i]
        sum_f1+=f1_list[i]
        sum_accuracy+=accuracy_list[i]
    average_precision=sum_precision/5
    average_recall=sum_recall/5
    average_f1=sum_f1/5
    average_accuracy=sum_accuracy/5
    print("average result:"+" precision: "+str(average_precision)+", recall: "+str(average_recall)+", f1: "+str(average_f1)+", accuracy:"+str(average_accuracy))
    record.write("average result:"+" precision: "+str(average_precision)+", recall: "+str(average_recall)+", f1: "+str(average_f1)+", accuracy:"+str(average_accuracy))
    record.close()
    end_time=time.time()
    print("prediction time:"+str(end_time-start_time))
    print("finish!")

if __name__ == '__main__':
    # main('/home/lhr/ccd/data/ast_llvm_semantic_40.csv')
    # main('/home/cy/ccd/Prism-training/data/ast_llvm_semantic_45.csv')
    main('/home/cy/ccd/dataset/GCJ_test_merge_trainingData/merge.csv')
    sys.exit(0)