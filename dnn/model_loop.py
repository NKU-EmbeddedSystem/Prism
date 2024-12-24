
import torch.nn as nn
import torch
import torch.nn.functional as F


class trans_layer(nn.Module):
    def __init__(self,feature_len):
        super(trans_layer,self).__init__()


        self.feature_len = feature_len


        self.conv_l1_1 = nn.Conv1d(1,1,kernel_size=3, padding=1)
        self.conv_l1_2 = nn.Conv1d(1,1,kernel_size=3, padding=1)
        self.conv_l1_3 = nn.Conv1d(1,1,kernel_size=3, padding=1)


        self.conv_l2_1 = nn.Conv1d(1,1,kernel_size=3, padding=1)
        self.conv_l2_2 = nn.Conv1d(1,1,kernel_size=3, padding=1)
        self.conv_l2_3 = nn.Conv1d(1,1,kernel_size=3, padding=1)


        self.lrelu = nn.LeakyReLU(0.1, True)

    
    def forward(self,x):

        '''
        s_1 : assembly code representaion from ARM
        s_2 : assembly code representaion from x86
        s_3 : source code representation from bert
        '''


        s_1 = x[..., :self.feature_len]
        s_2 = x[..., self.feature_len :self.feature_len*2]
        s_3 = x[..., self.feature_len*2:]
        
        x_1 = self.lrelu(self.conv_l1_1(s_1)) #feature_len * feature_num
        x_2 = self.lrelu(self.conv_l1_2(s_2))
        x_3 = self.lrelu(self.conv_l1_3(s_3))


        out_1 = self.lrelu(self.conv_l2_1(x_1)) 
        out_2 = self.lrelu(self.conv_l2_2(x_2)) 
        out_3 = self.lrelu(self.conv_l2_3(x_3)) 


        return out_1, out_2, out_3




class fusion_layer(nn.Module):
    def __init__(self,feature_len,kernel_size):
        super(fusion_layer,self).__init__()


        self.feature_len = feature_len
        self.kernel_size = kernel_size
        self.de_dimen = torch.nn.Linear(feature_len, kernel_size * kernel_size)
        self.pool = torch.nn.AvgPool1d(feature_len)

        self.conv_1 = nn.Conv1d(1,1,kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(1,1,kernel_size=3, padding=1)
        self.conv_3 = nn.Conv1d(1,1,kernel_size=3, padding=1)


        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.1, True)


    def forward(self, x):
        '''
        x_[0] : assembly code representaion from ARM
        x_[1] : assembly code representaion from x86
        x_[2] : source code representation from bert
        '''
        # print(x.size())
        b,c,l = x[0].size()

        #[0,1-2] [0,2-1] [1,2-0]

        arm_emb = x[0]
        x86_emb = x[1]
        bet_emb = x[2]

 

        #assembly code representaion fusion
        s_1_t= arm_emb.permute(0,2,1)
        matmul_s_12 = self.lrelu(torch.matmul(s_1_t, x86_emb))


        s_3_d = self.lrelu(self.de_dimen(bet_emb))
        fusion_embedding = self.lrelu(F.conv2d(matmul_s_12.view(1,-1,self.feature_len,self.feature_len),s_3_d.view(-1,1,self.kernel_size,self.kernel_size),groups=b*c,padding = (self.kernel_size-1)//2))

        fusion_embedding = fusion_embedding.view(-1,self.feature_len,self.feature_len)

        w_pool = self.pool(fusion_embedding)#b,96,1
        h_pool = self.pool(fusion_embedding.permute(0,2,1))#b,96,1

        output1 = w_pool.view(b,self.feature_len)
        output2 = h_pool.view(b,self.feature_len)

        output_embedding = torch.cat((output1, output2), dim=1)

        return output_embedding



class Classifier_Net(nn.Module):#设置网络
    def __init__(self, feature_len,  kernel_size, n_output):
        super(Classifier_Net, self).__init__()


        self.classifier_shallow = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_len*6, feature_len*3),
            nn.BatchNorm1d(feature_len*3),
            nn.ReLU(inplace=True),
            nn.Linear(feature_len*3, n_output))


        self.trans_layer = trans_layer(feature_len)   
        self.fusion_layer = fusion_layer(feature_len,kernel_size)
        self.reshape_layer = nn.Sequential(
            torch.nn.Linear(feature_len*2, feature_len*3),
            nn.BatchNorm1d(feature_len*3),
            nn.LeakyReLU(0.1, True))



      


    def forward(self, x):

        x = x.unsqueeze(dim=1)
        # print(x.size())

        code_1 = x[..., :288]#feature_len * feature_num
        code_1_init = self.trans_layer(code_1) 
        # # print(code_1.size()) 

        code_2 = x[..., 288:]
        code_2_init = self.trans_layer(code_2)

        
        code_1_emb = self.fusion_layer(code_1_init)
        code_2_emb = self.fusion_layer(code_2_init)


        code_1_emb = self.reshape_layer(code_1_emb)
        code_2_emb = self.reshape_layer(code_2_emb)

        code1_embedding = code_1_emb + code_1.squeeze(dim = 1)
        code2_embedding = code_2_emb + code_2.squeeze(dim = 1)


        code_embedding = torch.cat ( (code1_embedding, code2_embedding), dim=1)#torch.Size([512, 576])

    
        output_embedding = self.classifier_shallow(code_embedding)
        return output_embedding
