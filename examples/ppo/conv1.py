import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import math
class HGTConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads,dropout = 0.2, use_norm = True, use_RTE = False, **kwargs):
        super(HGTConv, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        self.WK = nn.Parameter(torch.empty(5, self.out_dim, self.out_dim))
        self.bK = nn.Parameter(torch.empty(5, self.out_dim, 1))
        self.WQ = nn.Parameter(torch.empty(5, self.out_dim, self.out_dim))
        self.bQ = nn.Parameter(torch.empty(5, self.out_dim, 1))
        self.WM = nn.Parameter(torch.empty(5, self.out_dim, self.out_dim))
        self.bM = nn.Parameter(torch.empty(5, self.out_dim, 1))
        self.WB = nn.Parameter(torch.empty(5, self.out_dim, self.out_dim))
        self.bB = nn.Parameter(torch.empty(5, self.out_dim, 1))
        self.Wmsgup = nn.Parameter(torch.empty(self.d_k, self.d_k))
        self.Wmsgdown = nn.Parameter(torch.empty(self.d_k, self.d_k))
        self.Wmsgleft = nn.Parameter(torch.empty(self.d_k, self.d_k))
        self.Wmsgright = nn.Parameter(torch.empty(self.d_k, self.d_k))
        xavier_uniform_(self.WK)
        xavier_uniform_(self.WQ)
        xavier_uniform_(self.WM)
        xavier_uniform_(self.WB)
        xavier_normal_(self.bK)
        xavier_normal_(self.bQ)
        xavier_normal_(self.bM)
        xavier_normal_(self.bB)
        xavier_normal_(self.Wmsgup)
        xavier_normal_(self.Wmsgdown)
        xavier_normal_(self.Wmsgleft)
        xavier_normal_(self.Wmsgright)
        #self.d_per_head = self.out_dim // n_heads
        #self.ME = nn.Parameter(torch.ones(self.n_heads,25,25))
        
        
    def forward(self, node_inp, WW,mask_matrix,R,attn_mask):
        d,batch_size,n = node_inp.shape
        m = 5 
        N = 15 
        a = self.WK.view(1,5, self.out_dim, self.out_dim)   
        WK = a.repeat(batch_size,1,1,1)    #key的映射矩阵#
        b = self.bK.view(1,5, self.out_dim, 1)   
        bK = b.repeat(batch_size,1,1,1)    #key的偏置#
        c = self.WQ.view(1,5, self.out_dim, self.out_dim)   
        WQ = c.repeat(batch_size,1,1,1)    #query的映射矩阵#
        
        TT = self.bQ.view(1,5, self.out_dim, 1)   
        bQ = TT.repeat(batch_size,1,1,1)   #query的偏置#
        e = self.WM.view(1,5, self.out_dim, self.out_dim)    
        WM = e.repeat(batch_size,1,1,1)    #message(即value)的映射矩阵#
        f = self.bM.view(1,5, self.out_dim, 1)   
        bM = f.repeat(batch_size,1,1,1)    #message的偏置#
        g = self.WB.view(1,5, self.out_dim, self.out_dim)   
        WB = g.repeat(batch_size,1,1,1)    #信息聚合后的矩阵B中对不同的节点种类事假不同的线性变换#
        h = self.bB.view(1,5, self.out_dim, 1)   
        bB = h.repeat(batch_size,1,1,1)    #信息聚合后的偏置#
        #I = self.ME.view(1,self.n_heads,25,25)   
        #ME = I.repeat(batch_size,1,1,1)    #先验张量μ的堆叠矩阵，以突显不同邻接关系的重要性#
        #print("node_inp:",node_inp)
        node_inp_m = node_inp.unsqueeze(1).repeat(1, m, 1, 1).permute(2, 1, 0, 3)   #将输入张量堆叠m次，此时(batch_size,m,d,n)#
        node_inp_m = torch.bmm(node_inp_m.reshape(batch_size * m, d, n), WW.reshape(batch_size * m, n, N))   #按节点种类划分节点，(batch_size,m,d,N)#
        K = torch.bmm(WK.reshape(batch_size * m, d, d),node_inp_m.reshape(batch_size * m, d, N)) + bK.reshape(batch_size * m, d, 1)
        K = K.reshape(batch_size,m,d,N).permute(0,2,1,3).reshape(batch_size,d,m*N)
        K = torch.matmul(K, mask_matrix)
        
        Q = torch.bmm(WQ.reshape(batch_size * m, d, d),node_inp_m.reshape(batch_size * m, d, N)) + bQ.reshape(batch_size * m, d, 1)
        Q = Q.reshape(batch_size,m,d,N).permute(0,2,1,3).reshape(batch_size,d,m*N)
        Q = torch.matmul(Q, mask_matrix)     #K,Q:(batch_size,d,n)#
        
        K = K.reshape(batch_size,self.n_heads,self.d_k,n)   
        Q = Q.reshape(batch_size,self.n_heads,self.d_k,n)   #(batch_size,h,d/h,n)#
        A=torch.matmul(Q.permute(0,1,3,2),K)/self.sqrt_dk 

        attention_weights = F.softmax(R+A, dim=3)
        attention_weights = torch.where(torch.isnan(attention_weights),torch.full_like(attention_weights,0),attention_weights)
        attention_weights = attention_weights.repeat(1,1,1,4)*attn_mask
        message = torch.bmm(WM.reshape(batch_size * m, d, d),node_inp_m.reshape(batch_size * m, d, N)) + bM.reshape(batch_size * m, d, 1)
        message = message.reshape(batch_size,d,m*N)
        message = torch.matmul(message, mask_matrix)
        res_msg = message.reshape(batch_size,self.n_heads,self.d_k,n)  #信息计算，(batch_size,h,d/h,n)#
        Wmsg = torch.cat((self.Wmsgup,self.Wmsgdown,self.Wmsgleft,self.Wmsgright),dim = 1)
        res_msg = torch.bmm(Wmsg.permute(1,0).unsqueeze(0).repeat(batch_size * self.n_heads,1,1),res_msg.reshape(batch_size * self.n_heads, self.d_k, n)).reshape(batch_size,self.n_heads,4*self.d_k,n)
        res_msg = res_msg.reshape(batch_size,self.n_heads,self.d_k,4*n)
        B=torch.bmm(attention_weights.reshape(batch_size * self.n_heads, n, 4*n),torch.transpose(res_msg.reshape(batch_size * self.n_heads, self.d_k, 4*n),dim0=1, dim1=2)).permute(0,2,1) 
        B = B.reshape(batch_size,d,n)  
        B = F.gelu(B)  
        B = B.unsqueeze(1).repeat(1, m, 1, 1)  #(batch_size,m,d,n)#
        B = torch.bmm(B.reshape(batch_size * m, d, n),WW.reshape(batch_size * m, n, N))
        B = torch.bmm(WB.reshape(batch_size * m, d, d),B.reshape(batch_size * m, d, N)) + bB.reshape(batch_size * m, d, 1)
        B = B.reshape(batch_size,m,d,N).permute(0,2,1,3).reshape(batch_size,d,m*N)
        B = torch.matmul(B, mask_matrix).permute(1,0,2)
        attention_weights = attention_weights[0,0,:,:]  
        output = B+node_inp 
        return output,attention_weights



    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm = True, use_RTE = True):
        super(GeneralConv, self).__init__()
        self.base_conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads,  dropout, use_norm, use_RTE)
    def forward(self, meta_xs, WW,mask_matrix,R,attn_mask):
        return self.base_conv(meta_xs,WW,mask_matrix,R,attn_mask)