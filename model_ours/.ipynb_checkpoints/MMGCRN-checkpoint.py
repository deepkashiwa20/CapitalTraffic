import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, node_embeddings):
        if len(node_embeddings.shape)==2:
            node_num = node_embeddings.shape[0]
            supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        else:
            node_num = node_embeddings.shape[1]
            supports = F.softmax(F.relu(torch.einsum('bnc,bmc->nm', node_embeddings, node_embeddings)), dim=1)            
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2]) 
        x_g = []
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, cheb_k * dim_in
        # supports = torch.stack(support_set)
        # x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_g = x_g.reshape(batch_num, node_num, -1) # B, N, cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class ADCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(ADCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        # return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class ADCRNN_STEP(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(ADCRNN_STEP, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], node_embeddings)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MMGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, embed_dim=8, cheb_k=3,
                 ycov_dim=1, mem_num=10, mem_dim=32, memory_type='local', meta_type='yes', decoder_type='stepwise', go_type='go'):
        super(MMGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        self.memory_type = memory_type
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        
        self.ycov_dim = ycov_dim # float type t_cov or history_cov
            
        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()
        
        if self.memory_type in ['local', 'double']:
            self.decoder_dim = self.hidden_dim + self.mem_dim # add historical average
        else:
            self.decoder_dim = self.hidden_dim
        self.meta_type = meta_type
        
        # encoder
        self.encoder = ADCRNN(num_nodes, self.input_dim, rnn_units, cheb_k, embed_dim, num_layers)     # mob
        
        # deocoder
        self.decoder_type = decoder_type
        self.go_type = go_type
        if self.decoder_type == 'sequence':
            self.decoder = ADCRNN(num_nodes, self.ycov_dim, self.decoder_dim, cheb_k, embed_dim, num_layers)     # mob
        elif self.decoder_type == 'stepwise':
            self.decoder = ADCRNN_STEP(num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, cheb_k, embed_dim, num_layers)  # mob
        else:
            self.decoder = None
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        
    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        if self.memory_type == 'local':
            memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
            memory_dict['Wq'] = nn.Parameter(torch.randn(self.hidden_dim, self.mem_dim), requires_grad=True)    # project to query
            memory_dict['FC_E'] = nn.Parameter(torch.randn(self.mem_dim, self.embed_dim), requires_grad=True)
        elif self.memory_type == 'double':
            memory_dict['Memory0'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d), first memory is normal, second is abnormal.
            memory_dict['Wq0'] = nn.Parameter(torch.randn(self.hidden_dim, self.mem_dim), requires_grad=True)      # project to query
            memory_dict['FC_E0'] = nn.Parameter(torch.randn(self.mem_dim, self.embed_dim), requires_grad=True)
            memory_dict['Memory1'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d), first memory is normal, second is abnormal.
            memory_dict['Wq1'] = nn.Parameter(torch.randn(self.hidden_dim, self.mem_dim), requires_grad=True)      # project to query
            memory_dict['FC_E1'] = nn.Parameter(torch.randn(self.mem_dim, self.embed_dim), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    

    def query_memory(self, h_t:torch.Tensor, ex_t:torch.Tensor=None):
        if self.memory_type == 'local':
            B = h_t.shape[0] # h_t = h_t.squeeze(1) # B, N, hidden
            query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
            att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
            proto_t = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)
            W_E = torch.matmul(proto_t, self.memory['FC_E']) # (B, N, e)
            _, ind = torch.topk(att_score, k=2, dim=-1)
            pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d
            neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
            return W_E, proto_t, query, pos, neg
        
        elif self.memory_type == 'double':
            B, N = h_t.shape[0], h_t.shape[1] # B, N, hidden
            ex_t = ex_t.int()
            W_E = torch.zeros((B, N, self.embed_dim), device=h_t.device)
            proto_t = torch.zeros((B, N, self.mem_dim), device=h_t.device)
            query = torch.zeros((B, N, self.mem_dim), device=h_t.device)
            pos = torch.zeros((B, N, self.mem_dim), device=h_t.device)
            neg = torch.zeros((B, N, self.mem_dim), device=h_t.device)
            
            h_t_normal, h_t_abnormal = h_t[ex_t==0], h_t[ex_t==1]
            
            if h_t_normal.shape[0] > 0:
                query_normal = torch.matmul(h_t_normal, self.memory['Wq0'])
                score_normal = torch.softmax(torch.matmul(query_normal, self.memory['Memory0'].t()), dim=-1)
                proto_normal = torch.matmul(score_normal, self.memory['Memory0'])
                W_E_normal = torch.matmul(proto_normal, self.memory['FC_E0'])
                _, ind_normal = torch.topk(score_normal, k=2, dim=-1)
                pos_normal = self.memory['Memory0'][ind_normal[:, 0]]
                neg_normal = self.memory['Memory0'][ind_normal[:, 1]]
                W_E[ex_t==0] = W_E_normal
                proto_t[ex_t==0] = proto_normal
                query[ex_t==0] = query_normal
                pos[ex_t==0] = pos_normal
                neg[ex_t==0] = neg_normal
                
            if h_t_abnormal.shape[0] > 0:
                query_abnormal = torch.matmul(h_t_abnormal, self.memory['Wq1'])
                score_abnormal = torch.softmax(torch.matmul(query_abnormal, self.memory['Memory1'].t()), dim=-1)
                proto_abnormal = torch.matmul(score_abnormal, self.memory['Memory1'])
                W_E_abnormal = torch.matmul(proto_abnormal, self.memory['FC_E1'])
                _, ind_abnormal = torch.topk(score_abnormal, k=2, dim=-1)
                pos_abnormal = self.memory['Memory1'][ind_abnormal[:, 0]]
                neg_abnormal = self.memory['Memory1'][ind_abnormal[:, 1]]
                W_E[ex_t==1] = W_E_abnormal
                proto_t[ex_t==1] = proto_abnormal
                query[ex_t==1] = query_abnormal
                pos[ex_t==1] = pos_abnormal
                neg[ex_t==1] = neg_abnormal
                
            return W_E, proto_t, query, pos, neg
        else:
            assert False, 'You must specify a correct memory type...'
            
    def forward(self, x, y_cov):
        if x.shape[-1] > 1 and self.input_dim == 1:
            x, ex_t = x[:, :, :, :1], x[:, -1, :, 1] # x and accident at t
        
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, self.node_embeddings)      # B, T, N, hidden      
        
        h_t = h_en[:, -1, :, :]                               # B, N, hidden (last state)        
        if self.memory_type == 'local':
            _node_embeddings, h_att, query, pos, neg = self.query_memory(h_t)
            h_t = torch.cat([h_t, h_att], dim=-1)            
        elif self.memory_type == 'double':
            assert ex_t is not None, 'ex_t does not exist, so please set --memory=double --incident=True --exchannel=0'
            _node_embeddings, h_att, query, pos, neg = self.query_memory(h_t, ex_t)
            h_t = torch.cat([h_t, h_att], dim=-1)
        else:
            _node_embeddings = None
        ht_list = [h_t]*self.num_layers
        
        if self.decoder_type == 'sequence':
            if self.meta_type == 'yes':
                assert _node_embeddings is not None, 'meta graph (node embedding) must derive from memory...'
                h_de, state_de = self.decoder(y_cov, ht_list, _node_embeddings)
            else:
                h_de, state_de = self.decoder(y_cov, ht_list, self.node_embeddings)
            output = self.proj(h_de)
        elif self.decoder_type == 'stepwise':
            if self.go_type == 'random':
                go = torch.zeros((x.shape[0], self.num_node, self.output_dim), device=x.device)
            elif self.go_type == 'last':
                go = x[:, -1, :, :self.output_dim] # using the last input value instead of random.
            else:
                assert False, 'You must specify a correct go type: random or last'
            out = []
            for t in range(self.horizon):
                if self.meta_type == 'yes':
                    assert _node_embeddings is not None, 'meta graph (node embedding) must derive from memory...'
                    h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, _node_embeddings)
                else:
                    h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, self.node_embeddings)
                go = self.proj(h_de)
                out.append(go)
            output = torch.stack(out, dim=1)
        else:
            assert False, 'You must specify a correct decoder type: sequence, step_go_ycov, step_ycov'
        
        return output, h_att, query, pos, neg

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f' \n In total: {param_count} trainable parameters. \n')
    return

def main():
    import sys
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3, help="which GPU to use")
    parser.add_argument('--his_len', type=int, default=6, help='sequence length of observed historical values')
    parser.add_argument('--seq_len', type=int, default=6, help='sequence length of values, which should be even nums (2,4,6,12)')
    parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
    parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
    parser.add_argument('--hiddenunits', type=int, default=32, help='number of hidden units')
    parser.add_argument("--memory", type=str, default='local', help="which type of memory: local or any other")
    parser.add_argument("--meta", type=str, default='yes', help="whether to use meta-graph: yes or any other")
    parser.add_argument("--decoder", type=str, default='stepwise', help="which type of decoder: stepwise or sequence")
    parser.add_argument('--go', type=str, default='last', help='which type of decoder go: random or last')
    parser.add_argument('--incident', type=bool, default=False, help='incident')
    parser.add_argument('--exchannel', type=int, default=0, help='how many extra/external channels')
    opt = parser.parse_args()
    opt.channelin += opt.exchannel
    
    T_DIM = 38
    num_variable = 1609
    
    device = torch.device("cuda:{}".format(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    model = MMGCRN(num_nodes=num_variable, input_dim=opt.channelin, output_dim=opt.channelout, horizon=opt.seq_len, rnn_units=opt.hiddenunits, 
                        memory_type=opt.memory, meta_type=opt.meta, decoder_type=opt.decoder, go_type=opt.go).to(device)
    print_params(model)
    tmp_channelin = opt.channelin if opt.channelin > 1 else opt.channelin + int(opt.incident)
    print('tmp_channelin', tmp_channelin)
    summary(model, [(opt.his_len, num_variable, tmp_channelin), (opt.seq_len, num_variable, opt.channelout)], device=device)
        
if __name__ == '__main__':
    main()
