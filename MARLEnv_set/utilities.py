print("using shaed utilities code")

import copy
from typing import Optional, Any, Union, Callable
from torch import nn
import torch
from torch import Tensor
from torch.nn  import functional as F
from torch.nn.modules import Module
from torch.nn import MultiheadAttention
from torch.nn.modules import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Dropout
from torch.nn.modules import Linear
from torch.nn.modules import LayerNorm

import math
######utilitis related to transformer and its data processing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arrange_slots_heads_forMechanisms(x,n_mechanisms,n_heads,TLength,start_symbol=2):
    ###x has the shape (T,bz)
    
    #####split the original input sequence in to N sequence , one for each of the N mechaisms.and pad them into the same length
    x=x.to(DEVICE)
    x=x[1:x.shape[0]]#remove BOS


    #mechaisms_len is the number of toekn supoesed to be in each mechanism
    mechaisms_len=TLength//n_mechanisms

    x=F.pad(input=x, pad=(0,0,0,TLength-x.shape[0]), mode='constant', value=1).to(DEVICE)

  
    xs=list(torch.split(x,mechaisms_len))
    ###add in BOS token to intput into each mechanisms
    for i in range(len(xs)):
        xs[i]=torch.cat([torch.zeros(1,x.shape[1]).fill_(start_symbol).type(torch.int).to(DEVICE),xs[i]],0)


    return xs,TLength

def rearrange_slots_heads(xs,n_mechanisms,n_heads,TLength,start_symbol=2):
        ###this function put the list of output from each mechansims back to the original shape for inter-mechanisms 
            
    for i in range(1,len(xs)):
        xs[i]=xs[i][1:xs[i].shape[0]]###remove the BOS for each mechanism, except for the first mechanism
    recontructed=torch.cat(xs,0).to(DEVICE)
   
    recontructed=recontructed[range(TLength+1),:,:]
    ###putback the BOS token at begining
    #recontructed=torch.cat([torch.zeros(1,recontructed.shape[1],1).fill_(start_symbol).type(torch.int),recontructed],0)


    return recontructed



def generate_square_subsequent_mask(sz):

    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask_originalTransformer(src, tgt,n_mechanisms,n_disretizationheads,PAD_IDX=1):
    ######the src and tgt are in the form index (not embeding yet)
     #####each segment may have different padding patterns
    #xs,_=arrange_slots_heads_forMechanisms(tgt,n_mechanisms,n_disretizationheads,start_symbol=2)
    #tgt=torch.cat(xs,0)###put it back into a tnesor

    if src!=None:
        src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    #src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    #memory_mask = torch.zeros((tgt_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    src_mask = None
    memory_mask = None

    if src!=None:
        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    else:
        src_padding_mask =None
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

  


    return src_mask,memory_mask, tgt_mask, src_padding_mask, tgt_padding_mask





def create_mask_factorTIM(src, tgt,n_mechanisms,n_disretizationheads,MechanismGap,PAD_IDX=1):
    ########has a list of factor mask, which only allows each factor attend to the same factor
    ######the src and tgt are in the form index (not embeding yet)
     #####each segment may have different padding patterns
    #xs,_=arrange_slots_heads_forMechanisms(tgt,n_mechanisms,n_disretizationheads,start_symbol=2)
    #tgt=torch.cat(xs,0)###put it back into a tnesor

    if src!=None:
        src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    #src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    #memory_mask = torch.zeros((tgt_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    src_mask = None
    memory_mask = None

    if src!=None:
        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        src_padding_mask = src_padding_mask.float().masked_fill(src_padding_mask == 1, float('-inf')).masked_fill(src_padding_mask == 0, float(0.0))

    else:
        src_padding_mask =None
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf')).masked_fill(tgt_padding_mask == 0, float(0.0))


    ####mask for each mechanism
    All_mechanism_tgt_mask=[]
    for m in range(MechanismGap):
        mechanism_tgt_mask=torch.zeros(tgt_mask.shape).float()
        mechanism_tgt_mask=mechanism_tgt_mask.masked_fill(mechanism_tgt_mask == 0, float('-inf'))####mask all
        mechanism_tgt_mask[:,0]=0####the first token (BOS) is unmasked  
        mechanism_tgt_mask[:,(1+m)::MechanismGap]=0 ####the same factors are not masked
        mechanism_tgt_mask[(torch.triu(torch.ones((mechanism_tgt_mask.shape[0], mechanism_tgt_mask.shape[1])),diagonal=1)) == 1]=float("-inf") ####upper triangle are masked to prevent attention into the future
        All_mechanism_tgt_mask.append(mechanism_tgt_mask)
    All_mechanism_tgt_mask=torch.stack(All_mechanism_tgt_mask,0)

    return src_mask,memory_mask, tgt_mask, src_padding_mask, tgt_padding_mask,All_mechanism_tgt_mask



def create_mask_TIM(src, tgt,n_mechanisms,n_disretizationheads,TLength,PAD_IDX=1):
    ######the src and tgt are in the form index (not embeding yet)
    

     #####each segment may have different padding patterns
    xs,_=arrange_slots_heads_forMechanisms(tgt,n_mechanisms,n_disretizationheads,TLength,start_symbol=2)

    tgt=torch.cat(xs,0)###put it back into a tnesor
    if src!=None:
        src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    #src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    #memory_mask = torch.zeros((tgt_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    src_mask = None
    memory_mask = None

    if src!=None:
        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    else:
        src_padding_mask =None 
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    #####memory mechanism mask are used for each quantization head or each slot shape (T//n_mechanisms+BOS,bz,E)
    #memory_mechanism_mask = torch.zeros((n_disretizationheads+1, src_seq_len),device=DEVICE).type(torch.bool)
    memory_mechanism_mask = None

    #####target mechanism mask are used for each quantization head or each slot shape (T//n_mechanisms,bz,E)
    mechaisms_len=TLength//n_mechanisms
    tgt_mechanism_mask=generate_square_subsequent_mask(mechaisms_len+1)


   
         
    tgt_mechanism_key_padding_masks=[]
    for j in range(n_mechanisms): 
        seg=(xs[j].clone().detach()==PAD_IDX).transpose(0, 1)
        tgt_mechanism_key_padding_masks.append(seg)
    tgt_mechanism_key_padding_masks=torch.stack(tgt_mechanism_key_padding_masks)###shape: (n_mechanism, T//n_mechanisms,bz,E)


    return src_mask,memory_mask, tgt_mask, src_padding_mask, tgt_padding_mask,memory_mechanism_mask,tgt_mechanism_mask,tgt_mechanism_key_padding_masks



class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # print("emb_size")
        # print(self.emb_size)
        # print(tokens.shape)
        # print(tokens.max())

        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])



###########arrange target seq in patch version 
def patch_token_rearrange(x,n_heads,n_patches,reverse=False):

    ######x of zie (T, bz), include (BOS) and EOS, T-2 must =n_heads*n_patches
    assert (x.shape[0]-2)==n_heads*n_patches


    if reverse==False:
        Vec=[]
        Vec.append(x[0,:].unsqueeze(0))#<BOS>
        x_truncated=x[1:-1,]
        for i in range(n_heads):
            #print(x_truncated[i::n_heads,])
            Vec.append(x_truncated[i::n_heads,])
        Vec.append(x[-1,:].unsqueeze(0))##EOS
        x=torch.cat(Vec,0)

    if reverse:

        Vec=[]
        Vec.append(x[0,:].unsqueeze(0))#<BOS>
        x_truncated=x[1:-1,]
        for i in range(n_patches):
            #print(x_truncated[i::n_patches,])
            Vec.append(x_truncated[i::n_patches,])
        Vec.append(x[-1,:].unsqueeze(0))##EOS
        x=torch.cat(Vec,0)

    return x


######function to convert image to patch and back 

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    nb = (h//block_size)*(w//block_size)
    return unfolded_x.reshape(n, c, block_size**2, h//block_size, w//block_size).permute(0,3,4,1,2).reshape((n * nb, c, block_size, block_size))

def depth_to_space(x, block_size, hb):
    n_hb,c,b,b = x.size()
    n = n_hb // (hb**2)
    x = x.reshape((n,hb,hb, c, block_size, block_size)).permute(0,3,4,5,1,2).reshape((n, c*block_size**2, hb, hb))
    return torch.nn.functional.pixel_shuffle(x, block_size)




####schedular optimizers

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr





#####performance checking 


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)

    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss






if __name__ == "__main__":
    print("testing the functions")


    src=None
    tgt=torch.zeros(64,404)
    n_mechanisms=4
    n_disretizationheads=4
    MechanismGap=4
    src_mask,memory_mask, tgt_mask, src_padding_mask, tgt_padding_mask,All_mechanism_tgt_mask=create_mask_factorTIM(src, tgt,n_mechanisms,n_disretizationheads,PAD_IDX=1,MechanismGap=MechanismGap)

    print("tgt_padding_mask")
    print(tgt_padding_mask)


    print("All_mechanism_tgt_mask")
    #print(All_mechanism_tgt_mask[0])
    #print(All_mechanism_tgt_mask[1])
    print(All_mechanism_tgt_mask[3][8,:])
    print(All_mechanism_tgt_mask[3].shape)
    print(All_mechanism_tgt_mask[3][-1,:])
    print(All_mechanism_tgt_mask[3].shape)

    print(All_mechanism_tgt_mask.shape)
    
    

