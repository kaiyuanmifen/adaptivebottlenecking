import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from quantize import Quantize

class QuantizerFunction(nn.Module):
    def __init__(self, input_dims, name, chkpt_dir,args,alpha=0.01,N_factors=[1,2,4],CodebookSize=96):
        super(QuantizerFunction, self).__init__()
        import os
        self.chkpt_file = os.path.join(chkpt_dir, name)
        
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

        # ###discretization function 

        self.args=args
        self.CodebookSize=CodebookSize


        self.N_factors=N_factors
        self.N_tightness_levels=len(self.N_factors)

        self.alpha=0.01###hyperparameter control penalizaing term for using more factors

        self.Quantization_projector=nn.Linear(input_dims, 8)#do projection so quantization can be done with mult-factors
        self.Quantization_projector_back=nn.Linear(8,input_dims)

        if "Adaptive" in args.Method:
            
          
            self.input_dims=input_dims

            self.QuantizeFunctions=nn.ModuleList([Quantize(8,self.CodebookSize//self.N_tightness_levels,N_factor) for N_factor in self.N_factors])   
            ###keys for the quantization modules 

            self.quantization_keys=torch.nn.Parameter(torch.randn(self.N_tightness_levels,1,8))

            self.quantization_attention=torch.nn.MultiheadAttention(embed_dim=8, num_heads=1)
        

        elif "Quantization"==args.Method:
            self.QuantizeFunctions=Quantize(8,self.CodebookSize,self.N_factors[0])   
            
            



        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
 

    def forward(self, state):

        if self.args.Method!="Original":
            state=self.Quantization_projector(state)

            bsz,Hsz=state.shape

            if self.args.Method=="Quantization":
                state=state.reshape(bsz,1,Hsz)
                state,CBloss,ind=self.QuantizeFunctions(state)#use a fixed function to discretize

                ExtraLoss=CBloss

                att_scores=torch.zeros(self.N_tightness_levels).unsqueeze(0).unsqueeze(1)

            elif self.args.Method=="Adaptive_Quantization":
                
                ###key-query attention to decide which quantization_function to use
                query=state.reshape(bsz,1,Hsz)
                
                _,att_scores=self.quantization_attention(query=query, key=self.quantization_keys,value=self.quantization_keys)

                att_scores=nn.functional.gumbel_softmax(att_scores,hard=True,dim=2)
        
                state=state.reshape(bsz,1,Hsz)

                Zs=[]
                CBloss=0
                for i in range(self.N_tightness_levels):
                    Z,CBloss_vec,_=self.QuantizeFunctions(state)#use a fixed function to discretize
                    Zs.append(Z)
                    CBloss+=CBloss_vec#sum up the codebookloss


                CBloss=CBloss/self.N_tightness_levels
                N_factor_vec=torch.tensor(self.N_factors).unsqueeze(0).unsqueeze(0).repeat(1,att_scores.shape[1],1).to(self.device)
                N_factor_penalty=(N_factor_vec*att_scores).mean(dim=1).sum() ###penalize the number of factors used

                ExtraLoss=CBloss+self.alpha*N_factor_penalty

                Zs=torch.cat(Zs,1)
        
                state=torch.bmm(att_scores.permute(1,0,2),Zs)

            elif self.args.Method=="Adaptive_Hierachical":
                
                ###key-query attention to decide which quantization_function to use
                query=state.reshape(bsz,1,Hsz)
                
                _,att_scores=self.quantization_attention(query=query, key=self.quantization_keys,value=self.quantization_keys)

                att_scores=nn.functional.gumbel_softmax(att_scores,hard=True,dim=2)

                
                state=state.reshape(bsz,1,Hsz)

                Zs=[]
                CBloss=0
                Z=state
                for i in range(self.N_tightness_levels):
                    Z,CBloss_vec,_=self.QuantizeFunctions[i](Z)#use a fixed function to discretize
                    Zs.append(Z)
                    CBloss+=CBloss_vec#sum up the codebookloss


                CBloss=CBloss/self.N_tightness_levels
                N_factor_vec=torch.tensor(self.N_factors).unsqueeze(0).unsqueeze(0).repeat(1,att_scores.shape[1],1).to(self.device)
                N_factor_penalty=(N_factor_vec*att_scores).mean(dim=1).sum() ###penalize the number of factors used

                ExtraLoss=CBloss+self.alpha*N_factor_penalty



                Zs=torch.cat(Zs,1)
        
                state=torch.bmm(att_scores.permute(1,0,2),Zs)

            state=state.reshape(bsz,Hsz)

            state=self.Quantization_projector_back(state)###shape it back


        elif self.args.Method=="Original":

            CBloss=0
            ExtraLoss=CBloss
            att_scores=torch.zeros(self.N_tightness_levels).unsqueeze(0).unsqueeze(1)

        return state,ExtraLoss,att_scores

    def save_models(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_models(self):
        self.load_state_dict(T.load(self.chkpt_file))