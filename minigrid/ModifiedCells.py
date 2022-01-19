import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp

from ShareParameters import	SharedParameters

class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class GroupLinearLayer(nn.Module):

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)


class GroupLinearLayer_with_shared_parameters(nn.Module):
	'''
	group linearlayers with share central pool parameters of schemas
	'''

	def __init__(self, device,din, dout, num_units,schema_weighting,num_schemas,Number_active):
		self.device=device
		super(GroupLinearLayer_with_shared_parameters,self).__init__()
		#generate shared parameter sets by units
		self.SharedParameters=SharedParameters(self.device,din, dout, num_schemas)
		self.SharedParameters.reset_schema_weighting(num_units,schema_weighting,Number_active)

		self.num_units=num_units
		self.num_schemas=num_schemas
		self.Number_active=Number_active
		#self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

	def forward(self, x):

		Weights = []

		for unit_idx in range(self.num_units):
			w, b = self.SharedParameters.get_weights_and_bias(unit_idx)
			Weights.append(w.unsqueeze(2))
		self.w = torch.cat(Weights, 2).permute(2, 1, 0)
		x = x.permute(1, 0, 2)

		x = torch.bmm(x, self.w)
		return x.permute(1, 0, 2)

class GroupLSTMCell(nn.Module):
	"""
	GroupLSTMCell can compute the operation of N LSTM Cells at once.
	"""
	def __init__(self, inp_size, hidden_size, num_lstms):
		super().__init__()
		self.inp_size = inp_size
		self.hidden_size = hidden_size
		
		self.i2h = GroupLinearLayer(inp_size, 4 * hidden_size, num_lstms)
		self.h2h = GroupLinearLayer(hidden_size, 4 * hidden_size, num_lstms)
		self.reset_parameters()


	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)

	def forward(self, x, hid_state):
		"""
		input: x (batch_size, num_lstms, input_size)
			   hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
		output: h (batch_size, num_lstms, hidden_state)
				c ((batch_size, num_lstms, hidden_state))
		"""
		h, c = hid_state
		preact = self.i2h(x) + self.h2h(h)

		gates = preact[:, :,  :3 * self.hidden_size].sigmoid()
		g_t = preact[:, :,  3 * self.hidden_size:].tanh()
		i_t = gates[:, :,  :self.hidden_size]
		f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
		o_t = gates[:, :, -self.hidden_size:]

		c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t) 
		h_t = torch.mul(o_t, c_t.tanh())

		return h_t, c_t


class GroupLSTMCell_sharedParameters(nn.Module):
	"""
	GroupLSTMCell can compute the operation of N LSTM Cells at once.
	And with shared parameter pool of schemas
	"""

	def __init__(self,device,input_size, hidden_size, num_lstms,schema_weighting,num_schemas,Number_active):
		self.device=device
		
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.x2h = GroupLinearLayer_with_shared_parameters(self.device,input_size, 4 * hidden_size, num_lstms,
														   schema_weighting[0], num_schemas, Number_active).to(self.device)
		self.h2h = GroupLinearLayer_with_shared_parameters(self.device,hidden_size, 4 * hidden_size, num_lstms,
														   schema_weighting[1], num_schemas, Number_active).to(self.device)



	def forward(self, x, hid_state):
		"""
		input: x (batch_size, num_lstms, input_size)
			   hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
		output: h (batch_size, num_lstms, hidden_state)
				c ((batch_size, num_lstms, hidden_state))
		"""
		h, c = hid_state
		preact = self.x2h(x) + self.h2h(h)

		gates = preact[:, :, :3 * self.hidden_size].sigmoid()
		g_t = preact[:, :, 3 * self.hidden_size:].tanh()
		i_t = gates[:, :, :self.hidden_size]
		f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
		o_t = gates[:, :, -self.hidden_size:]

		c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
		h_t = torch.mul(o_t, c_t.tanh())

		return h_t, c_t


class GroupGRUCell(nn.Module):
    """
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """
    def __init__(self, input_size, hidden_size, num_grus):
        super(GroupGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = GroupLinearLayer(input_size, 3 * hidden_size, num_grus)
        self.h2h = GroupLinearLayer(hidden_size, 3 * hidden_size, num_grus)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data = torch.ones(w.data.size())#.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """
		input: x (batch_size, num_grus, input_size)
			   hidden (batch_size, num_grus, hidden_size)
		output: hidden (batch_size, num_grus, hidden_size)
        """
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
        
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy


class GroupGRUCell_sharedParameters(nn.Module):
	"""
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """

	def __init__(self, device, input_size, hidden_size, num_grus,schema_weighting,num_schemas,Number_active):
		self.device=device

		super(GroupGRUCell_sharedParameters, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		#lauch the two layers with shared parameters
		self.x2h=GroupLinearLayer_with_shared_parameters(self.device, input_size, 3 * hidden_size, num_grus,
																		  schema_weighting[0],num_schemas,Number_active).to(self.device)
		self.h2h= GroupLinearLayer_with_shared_parameters(self.device, hidden_size, 3 * hidden_size, num_grus,
																		   schema_weighting[1],num_schemas,Number_active).to(self.device)


	# 	self.reset_parameters()
	#
	# def reset_parameters(self):
	# 	std = 1.0 / math.sqrt(self.hidden_size)
	# 	for w in self.parameters():
	# 		w.data = torch.ones(w.data.size())  # .uniform_(-std, std)

	def forward(self, x, hidden):
		"""
        input: x (batch_size, num_grus, input_size)
               hidden (batch_size, num_grus, hidden_size)
        output: hidden (batch_size, num_grus, hidden_size)
        """
		gate_x = self.x2h(x)
		gate_h = self.h2h(hidden)

		i_r, i_i, i_n = gate_x.chunk(3, 2)
		h_r, h_i, h_n = gate_h.chunk(3, 2)

		resetgate = torch.sigmoid(i_r + h_r)
		inputgate = torch.sigmoid(i_i + h_i)
		newgate = torch.tanh(i_n + (resetgate * h_n))

		hy = newgate + inputgate * (hidden - newgate)

		return hy



if __name__ == "__main__":

	GLN = GroupLinearLayer(12,64,25)

	x = torch.randn(64,25,12)

	print(GLN(x).shape)

	for p in GLN.parameters():
		print(p.shape)


	################group linear layer with shared parameters###########
	din=128
	dout=64
	num_units=6
	num_schemas=10
	Number_active=3
	schema_weighting=torch.randn((num_units, num_schemas), requires_grad=False)

	GLN=GroupLinearLayer_with_shared_parameters(din, dout, num_units, schema_weighting, num_schemas,Number_active)


	x = torch.randn(dout,num_units,din)

	print(GLN(x).shape)

	for p in GLN.parameters():
		print(p.shape)

	BatchSize=30
	num_units=6
	din=128
	x = torch.randn(BatchSize, num_units, din)
	y = torch.randn(BatchSize, num_units, dout)
	optimizer = torch.optim.Adam(GLN.parameters(), lr=0.0005)

	torch.autograd.set_detect_anomaly(True)
	optimizer.zero_grad()
	output = GLN(x)
	GLN.SharedParameters.schema_weighting[1]
	loss_fn = nn.MSELoss()
	loss = loss_fn(output, y)
	loss.backward(retain_graph=True)
	optimizer.step()

	##############grouped GRU cell with shared parameters#################
	din = 128
	hiddenSize = 100
	num_units = 6
	num_schemas = 10
	Number_active = 3
	schema_weighting = torch.randn((num_units, num_schemas), requires_grad=False)

	SGRU=GroupGRUCell_sharedParameters(din ,hiddenSize, num_units,
									   schema_weighting,num_schemas,Number_active)

	BatchSize=100
	inp = torch.randn(BatchSize,num_units, din)
	h = torch.randn(BatchSize,num_units,hiddenSize)

	h2= SGRU(inp,h)

	print('h2 shape', h2.shape)


	for parameter in SGRU.parameters():
		print(type(parameter.data), parameter.size())


	x= torch.randn(BatchSize,num_units, din)
	y=torch.randn(BatchSize, num_units, hiddenSize)

	optimizer = torch.optim.Adam(SGRU.parameters(), lr=0.0005)

	optimizer.zero_grad()
	output= SGRU(x,h)
	loss_fn=nn.MSELoss()
	loss = loss_fn(output, y)
	loss.backward(retain_graph=True)
	optimizer.step()

	##############grouped LSTM cell with shared parameters#################
	din = 128
	hiddenSize = 100
	num_units = 6
	num_schemas = 10
	Number_active = 3
	schema_weighting = torch.randn((num_units, num_schemas), requires_grad=False)

	SLSTM = GroupLSTMCell_sharedParameters(din, hiddenSize, num_units,
										 schema_weighting, num_schemas, Number_active)

	BatchSize = 100
	inp = torch.randn(BatchSize, num_units, din)
	h_c = (torch.randn(BatchSize, num_units, hiddenSize),torch.randn(BatchSize, num_units, hiddenSize))

	h_c = SLSTM(inp, h_c)

	print('h shape', h_c[0].shape)
	print('c shape', h_c[1].shape)

	for parameter in SLSTM.parameters():
		print(type(parameter.data), parameter.size())

	x = torch.randn(BatchSize, num_units, din)
	y = (torch.randn(BatchSize, num_units, hiddenSize),
		 torch.randn(BatchSize, num_units, hiddenSize))

	optimizer = torch.optim.Adam(SLSTM.parameters(), lr=0.0005)

	optimizer.zero_grad()
	output = SLSTM(x, h_c)
	loss_fn = nn.MSELoss()
	loss = loss_fn(output[1], y[1])
	loss.backward(retain_graph=True)
	optimizer.step()