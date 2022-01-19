import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from quantize import Quantize

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


# #############actor with quantization


# class ActorNetwork(nn.Module):
#     def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
#                  n_actions, name, chkpt_dir,args):
#         super(ActorNetwork, self).__init__()
#         import os
#         self.chkpt_file = os.path.join(chkpt_dir, name)


        
#         if not os.path.exists(chkpt_dir):
#             os.makedirs(chkpt_dir)

#         self.fc1 = nn.Linear(input_dims, fc1_dims)
#         self.fc2 = nn.Linear(fc1_dims, fc2_dims)
#         self.pi = nn.Linear(fc2_dims, n_actions)

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
#         self.to(self.device)


#         # ###discretization function 

#         self.args=args
#         if args!="Original":
#             self.N_tightness_levels=3
#             self.CodebookSize=16

          
#             self.input_dims=input_dims
#             self.Quantization_projector=nn.Linear(input_dims, 16)#do projection so quantization can be done with mult-factors
#             self.Quantization_projector_back=nn.Linear(16,input_dims)

#             self.QuantizeFunctions=nn.ModuleList([Quantize(16,self.CodebookSize,int(4//(2**j))) for j in range(self.N_tightness_levels)])   
#             ###keys for the quantization modules 

#             self.quantization_keys=torch.nn.Parameter(torch.randn(self.N_tightness_levels,1,16))

#             self.quantization_attention=torch.nn.MultiheadAttention(embed_dim=16, num_heads=4)



#         self.to(self.device)



#     def forward(self, state):

#         if self.args.Method!="Original":
#             state=self.Quantization_projector(state)

#             bsz,Hsz=state.shape

#             if self.args.Method=="Quantization":
#                 state=state.reshape(bsz,1,Hsz)
#                 state,CBloss,ind=self.QuantizeFunctions[1](state)#use a fixed function to discretize


#                 att_scores=torch.zeros(self.N_tightness_levels).unsqueeze(0).unsqueeze(1)

#             elif self.args.Method=="Adaptive_Quantization":
                
#                 ###key-query attention to decide which quantization_function to use
#                 query=state.reshape(bsz,1,Hsz)
                
#                 _,att_scores=self.quantization_attention(query=query, key=self.quantization_keys,value=self.quantization_keys)

#                 att_scores=nn.functional.gumbel_softmax(att_scores,hard=True,dim=2)
        
#                 state=state.reshape(bsz,1,Hsz)

#                 Zs=[]
#                 CBloss=0
#                 for i in range(self.N_tightness_levels):
#                     Z,CBloss_vec,_=self.QuantizeFunctions[i](state)#use a fixed function to discretize
#                     Zs.append(Z)
#                     CBloss+=CBloss_vec#sum up the codebookloss


#                 CBloss=CBloss/self.N_tightness_levels
#                 Zs=torch.cat(Zs,1)
        
#                 state=torch.bmm(att_scores.permute(1,0,2),Zs)

#             elif self.args.Method=="Adaptive_Hierachical":
                
#                 ###key-query attention to decide which quantization_function to use
#                 query=state.reshape(bsz,1,Hsz)
                
#                 _,att_scores=self.quantization_attention(query=query, key=self.quantization_keys,value=self.quantization_keys)

#                 att_scores=nn.functional.gumbel_softmax(att_scores,hard=True,dim=2)

                
#                 state=state.reshape(bsz,1,Hsz)

#                 Zs=[]
#                 CBloss=0
#                 Z=state
#                 for i in range(self.N_tightness_levels):
#                     Z,CBloss_vec,_=self.QuantizeFunctions[i](Z)#use a fixed function to discretize
#                     Zs.append(Z)
#                     CBloss+=CBloss_vec#sum up the codebookloss


#                 CBloss=CBloss/self.N_tightness_levels
#                 Zs=torch.cat(Zs,1)
        
#                 state=torch.bmm(att_scores.permute(1,0,2),Zs)

#             state=state.reshape(bsz,Hsz)

#             state=self.Quantization_projector_back(state)###shape it back


#         elif self.args.Method=="Original":

#             CBloss=0

#             att_scores=torch.zeros(self.N_tightness_levels).unsqueeze(0).unsqueeze(1)



        


#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         pi = T.softmax(self.pi(x), dim=1)

#         return pi,CBloss,att_scores

#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.chkpt_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.chkpt_file))