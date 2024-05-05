from .conv1 import *
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers,dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = False):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.pos_embedding = PositionalEncoding(d_model = n_hid, seq_len = 25)
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):   
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        #self.layers = GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE)
        for l in range(n_layers - 1): 
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))


    def forward(self, node_feature, WW,mask_matrix,R,attn_mask,node_type,change,rechange):
        res = torch.zeros(node_feature.size(0), node_feature.size(1),self.n_hid).to(node_feature.device)
        for t_id in range(5):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            idx = torch.unsqueeze(idx, dim=0)
            idx = idx.repeat(node_feature.size(1),1)
            idx = idx.reshape(-1,node_feature.size(1))         
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))   
        # meta_xs = self.drop(res)
        # del res
        attention_maps = []
        # meta_xs = self.drop(res)   
        meta_xs = res[rechange]
        del res
        meta_xs = self.pos_embedding(meta_xs)
        meta_xs = meta_xs[change.tolist()]
        meta_xs = meta_xs.permute(2, 1, 0) #(128,batch_size,25)#
        for gc in self.gcs: 
            meta_xs, attention_map = gc(meta_xs,WW,mask_matrix,R,attn_mask)
            attention_maps.append(attention_map) 
        return meta_xs, attention_maps  
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)