import torch.nn.functional as F
import torch
import torch.nn as nn
import math



class SharedParameters(nn.Module):

    def __init__(self,device, channels_in, channels_out, num_schemas):
        self.device=device

        super(SharedParameters, self).__init__()

        kernel_size = 1

        self.all_weight = nn.Parameter(torch.randn((num_schemas, channels_in, channels_out)))#be very careful here when mapping parameters between CPU/GPU the values might not be saved
        self.all_bias = nn.Parameter(torch.randn((num_schemas, channels_out)))

        self.num_schemas = num_schemas#Number of schemas per RIM unit
        #self.num_units=num_units #number of RIM units



    def reset_schema_weighting(self, num_units,schema_weighting=None,Number_active=3):
        # weighting of each schema in each units
        self.unm_units=num_units #number of RIM units
        self.Number_active=Number_active

        if schema_weighting is None:
            self.schema_weighting = nn.Parameter(torch.abs(torch.randn(num_units, self.num_schemas)), requires_grad=True)
        else:
            self.schema_weighting = nn.Parameter(schema_weighting, requires_grad=True)

        #print('schema weighting', schema_weighting)

        #print('schema weighting', schema_weighting)




    def get_weights_and_bias(self, unit_idx):

        ###maximum number active units, only top Number_active can be active, others aet set to zeros
        sel_schema_weights = self.schema_weighting[unit_idx]
        Filter = torch.zeros(self.num_schemas).to(self.device)
        Filter[torch.topk(sel_schema_weights, self.Number_active)[1]] = 1
        sel_schema_weights=sel_schema_weights*Filter
        
        sel_weight = torch.einsum('i,ijk->kj', sel_schema_weights,
                                  self.all_weight)  # weighted average of all schema paramaters

        sel_bias = torch.einsum('i,ik->k', sel_schema_weights, self.all_bias)  # weighted average of all biases


        return sel_weight,sel_bias

if __name__ == "__main__":
    mpc = SharedParameters(channels_in = 128, channels_out = 64, num_schemas =10)
    mpc.all_weight.shape
    mpc.all_bias.shape

    num_units=6
    num_schemas=10
    Number_active=3
    schema_weighting=torch.randn((num_units,num_schemas))
    mpc.reset_schema_weighting(num_units=num_units,Number_active=Number_active,schema_weighting=schema_weighting)

    for parameter in mpc.parameters():
        print(parameter.shape)


    x = torch.randn(32,128)
    res_lst = []

    for unit_idx in range(0,num_units):
        w,b = mpc.get_weights_and_bias(unit_idx)
        res = F.linear(x,w,bias=b)
        res_lst.append(res)


    Weights = []
    for unit_idx in range(num_units):
        w, b = mpc.get_weights_and_bias(unit_idx)
        Weights.append(w.unsqueeze(2))
    torch.cat(Weights, 2).permute(2,1,0).shape

    #try back prop
    ExternalGradient=[]
    for parameter in mpc.parameters():
        print(parameter.size())
        ExternalGradient.append(torch.randn(parameter.size()))


    optimizer = torch.optim.SGD(mpc.parameters(), lr=0.01, momentum=0.9)

    #
    x=torch.abs(torch.randn(num_units,num_schemas))
    Filter=torch.zeros(num_units,num_schemas)
    for i in range(num_units):
        Filter[i,torch.topk(x,3,dim=1)[1][i]]=1
    x=x*Filter
    x=x/torch.sum(x,dim=1,keepdim=True).repeat(1,num_schemas)
    torch.sum(x,1)