import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp

from ShareParameters import	SharedParameters
from ModifiedCells import *

class RIMCell(nn.Module):
	def __init__(self, 
		device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size = 64, input_value_size = 400, input_query_size = 64,
		num_input_heads = 1, input_dropout = 0.1, comm_key_size = 32, comm_value_size = 100, comm_query_size = 32, num_comm_heads = 4, comm_dropout = 0.1
	):
		super().__init__()
		if comm_value_size != hidden_size:
			#print('INFO: Changing communication value size to match hidden_size')
			comm_value_size = hidden_size
		self.device = device
		self.hidden_size = hidden_size
		self.num_units =num_units
		self.rnn_cell = rnn_cell
		self.key_size = input_key_size
		self.k = k
		self.num_input_heads = num_input_heads
		self.num_comm_heads = num_comm_heads
		self.input_key_size = input_key_size
		self.input_query_size = input_query_size
		self.input_value_size = input_value_size

		self.comm_key_size = comm_key_size
		self.comm_query_size = comm_query_size
		self.comm_value_size = comm_value_size

		self.key = nn.Linear(input_size, num_input_heads * input_query_size).to(self.device)
		self.value = nn.Linear(input_size, num_input_heads * input_value_size).to(self.device)

		if self.rnn_cell == 'GRU':
			self.rnn = GroupGRUCell(input_value_size, hidden_size, num_units)
			self.query = GroupLinearLayer(hidden_size,  input_key_size * num_input_heads, self.num_units)
		else:
			self.rnn = GroupLSTMCell(input_value_size, hidden_size, num_units)
			self.query = GroupLinearLayer(hidden_size,  input_key_size * num_input_heads, self.num_units)
		self.query_ =GroupLinearLayer(hidden_size, comm_query_size * num_comm_heads, self.num_units) 
		self.key_ = GroupLinearLayer(hidden_size, comm_key_size * num_comm_heads, self.num_units)
		self.value_ = GroupLinearLayer(hidden_size, comm_value_size * num_comm_heads, self.num_units)
		self.comm_attention_output = GroupLinearLayer(num_comm_heads * comm_value_size, comm_value_size, self.num_units)
		self.comm_dropout = nn.Dropout(p =input_dropout)
		self.input_dropout = nn.Dropout(p =comm_dropout)


	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
	    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
	    x = x.view(*new_x_shape)
	    return x.permute(0, 2, 1, 3)

	def input_attention_mask(self, x, h):
	    """
	    Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
	    		h (batch_size, num_units, hidden_size)
	    Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
	    		mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
		"""
	    key_layer = self.key(x)
	    value_layer = self.value(x)
	    query_layer = self.query(h)

	    key_layer = self.transpose_for_scores(key_layer,  self.num_input_heads, self.input_key_size)
	    value_layer = torch.mean(self.transpose_for_scores(value_layer,  self.num_input_heads, self.input_value_size), dim = 1)
	    query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)

	    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size) 
	    attention_scores = torch.mean(attention_scores, dim = 1)
	    mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)

	    not_null_scores = attention_scores[:,:, 0]
	    topk1 = torch.topk(not_null_scores,self.k,  dim = 1)
	    row_index = np.arange(x.size(0))
	    row_index = np.repeat(row_index, self.k)

	    mask_[row_index, topk1.indices.view(-1)] = 1
	    
	    attention_probs = self.input_dropout(nn.Softmax(dim = -1)(attention_scores))
	    inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)

	    return inputs, mask_

	def communication_attention(self, h, mask):
	    """
	    Input : h (batch_size, num_units, hidden_size)
	    	    mask obtained from the input_attention_mask() function
	    Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
	    """
	    query_layer = []
	    key_layer = []
	    value_layer = []
	    
	    query_layer = self.query_(h)
	    key_layer = self.key_(h)
	    value_layer = self.value_(h)

	    query_layer = self.transpose_for_scores(query_layer, self.num_comm_heads, self.comm_query_size)
	    key_layer = self.transpose_for_scores(key_layer, self.num_comm_heads, self.comm_key_size)
	    value_layer = self.transpose_for_scores(value_layer, self.num_comm_heads, self.comm_value_size)
	    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
	    attention_scores = attention_scores / math.sqrt(self.comm_key_size)
	    
	    attention_probs = nn.Softmax(dim=-1)(attention_scores)
	    
	    mask = [mask for _ in range(attention_probs.size(1))]
	    mask = torch.stack(mask, dim = 1)
	    
	    attention_probs = attention_probs * mask.unsqueeze(3)
	    attention_probs = self.comm_dropout(attention_probs)
	    context_layer = torch.matmul(attention_probs, value_layer)
	    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
	    new_context_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
	    context_layer = context_layer.view(*new_context_layer_shape)
	    context_layer = self.comm_attention_output(context_layer)
	    context_layer = context_layer + h
	    
	    return context_layer

	def forward(self, x, hs, cs = None):
		"""
		Input : x (batch_size, 1 , input_size)
				hs (batch_size, num_units, hidden_size)
				cs (batch_size, num_units, hidden_size)
		Output: new hs, cs for LSTM
				new hs for GRU
		"""
		size = x.size()
		null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
		x = torch.cat((x, null_input), dim = 1)

		# Compute input attention
		inputs, mask = self.input_attention_mask(x, hs)
		h_old = hs * 1.0
		if cs is not None:
			c_old = cs * 1.0
		

		# Compute RNN(LSTM or GRU) output
		
		if cs is not None:
			hs, cs = self.rnn(inputs, (hs, cs))
		else:
			hs = self.rnn(inputs, hs)

		# Block gradient through inactive units
		mask = mask.unsqueeze(2)
		h_new = blocked_grad.apply(hs, mask)

		# Compute communication attention
		h_new = self.communication_attention(h_new, mask.squeeze(2))

		hs = mask * h_new + (1 - mask) * h_old
		if cs is not None:
			cs = mask * cs + (1 - mask) * c_old
			return hs, cs

		return hs, None


class RIMCell_SelectiveActivation(nn.Module):
	def __init__(self,
				 device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size=64, input_value_size=400,
				 input_query_size=64,
				 num_input_heads=1, input_dropout=0.1, comm_key_size=32, comm_value_size=100, comm_query_size=32,
				 num_comm_heads=4, comm_dropout=0.1,
				 UnitActivityMask=None
				 ):
		super().__init__()
		if comm_value_size != hidden_size:
			# print('INFO: Changing communication value size to match hidden_size')
			comm_value_size = hidden_size
		self.device = device
		self.hidden_size = hidden_size
		self.num_units = num_units
		self.rnn_cell = rnn_cell
		self.key_size = input_key_size
		self.k = k
		self.num_input_heads = num_input_heads
		self.num_comm_heads = num_comm_heads
		self.input_key_size = input_key_size
		self.input_query_size = input_query_size
		self.input_value_size = input_value_size

		self.comm_key_size = comm_key_size
		self.comm_query_size = comm_query_size
		self.comm_value_size = comm_value_size

		self.key = nn.Linear(input_size, num_input_heads * input_query_size).to(self.device)
		self.value = nn.Linear(input_size, num_input_heads * input_value_size).to(self.device)

		if self.rnn_cell == 'GRU':
			self.rnn = GroupGRUCell(input_value_size, hidden_size, num_units)
			self.query = GroupLinearLayer(hidden_size, input_key_size * num_input_heads, self.num_units)
		else:
			self.rnn = GroupLSTMCell(input_value_size, hidden_size, num_units)
			self.query = GroupLinearLayer(hidden_size, input_key_size * num_input_heads, self.num_units)
		self.query_ = GroupLinearLayer(hidden_size, comm_query_size * num_comm_heads, self.num_units)
		self.key_ = GroupLinearLayer(hidden_size, comm_key_size * num_comm_heads, self.num_units)
		self.value_ = GroupLinearLayer(hidden_size, comm_value_size * num_comm_heads, self.num_units)
		self.comm_attention_output = GroupLinearLayer(num_comm_heads * comm_value_size, comm_value_size, self.num_units)
		self.comm_dropout = nn.Dropout(p=input_dropout)
		self.input_dropout = nn.Dropout(p=comm_dropout)

		##########unity activation mask, 1 means active 0 mean inactive#######
		if UnitActivityMask is None:
			print ("UnitActivityMask not in use ")
			self.UnitActivityMask=np.repeat(1,self.num_units )
		else:
			self.UnitActivityMask =UnitActivityMask

			print("using UnitActivityMask")
			print(self.UnitActivityMask)
		self.UnitActivityMask =torch.from_numpy(self.UnitActivityMask).to(self.device)


	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
		new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def input_attention_mask(self, x, h):
		"""
        Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
                h (batch_size, num_units, hidden_size)
        Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
                mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
        """
		key_layer = self.key(x)
		value_layer = self.value(x)
		query_layer = self.query(h)

		key_layer = self.transpose_for_scores(key_layer, self.num_input_heads, self.input_key_size)
		value_layer = torch.mean(self.transpose_for_scores(value_layer, self.num_input_heads, self.input_value_size),
								 dim=1)
		query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size)
		attention_scores = torch.mean(attention_scores, dim=1)

		#######setting attention score zeros for inactive units######
		attention_scores=attention_scores*self.UnitActivityMask
		print("input attention scores:")
		print(attention_scores)

		mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)

		not_null_scores = attention_scores[:, :, 0]
		topk1 = torch.topk(not_null_scores, self.k, dim=1)
		row_index = np.arange(x.size(0))
		row_index = np.repeat(row_index, self.k)

		mask_[row_index, topk1.indices.view(-1)] = 1

		attention_probs = self.input_dropout(nn.Softmax(dim=-1)(attention_scores))
		inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)

		return inputs, mask_

	def communication_attention(self, h, mask):
		"""
        Input : h (batch_size, num_units, hidden_size)
                mask obtained from the input_attention_mask() function
        Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
        """
		query_layer = []
		key_layer = []
		value_layer = []

		query_layer = self.query_(h)
		key_layer = self.key_(h)
		value_layer = self.value_(h)

		query_layer = self.transpose_for_scores(query_layer, self.num_comm_heads, self.comm_query_size)
		key_layer = self.transpose_for_scores(key_layer, self.num_comm_heads, self.comm_key_size)
		value_layer = self.transpose_for_scores(value_layer, self.num_comm_heads, self.comm_value_size)
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.comm_key_size)

		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		mask = [mask for _ in range(attention_probs.size(1))]
		mask = torch.stack(mask, dim=1)

		attention_probs = attention_probs * mask.unsqueeze(3)
		attention_probs = self.comm_dropout(attention_probs)
		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		context_layer = self.comm_attention_output(context_layer)
		context_layer = context_layer + h

		return context_layer

	def forward(self, x, hs, cs=None):
		"""
		Input : x (batch_size, 1 , input_size)
				hs (batch_size, num_units, hidden_size)
				cs (batch_size, num_units, hidden_size)
		Output: new hs, cs for LSTM
				new hs for GRU
		"""
		size = x.size()
		null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
		x = torch.cat((x, null_input), dim=1)

		# Compute input attention
		inputs, mask = self.input_attention_mask(x, hs)
		h_old = hs * 1.0
		if cs is not None:
			c_old = cs * 1.0

		# Compute RNN(LSTM or GRU) output

		if cs is not None:
			hs, cs = self.rnn(inputs, (hs, cs))
		else:
			hs = self.rnn(inputs, hs)

		# Block gradient through inactive units
		mask = mask.unsqueeze(2)
		h_new = blocked_grad.apply(hs, mask)

		# Compute communication attention
		h_new = self.communication_attention(h_new, mask.squeeze(2))

		hs = mask * h_new + (1 - mask) * h_old
		if cs is not None:
			cs = mask * cs + (1 - mask) * c_old
			return hs, cs

		return hs, None




class RIMCell_SharedParameters(nn.Module):
	def __init__(self,
				 device, input_size, hidden_size, num_units, k, rnn_cell,
				 schema_weighting, num_schemas,
				 input_key_size=64, input_value_size=400,
				 input_query_size=64,
				 num_input_heads=1, input_dropout=0.1, comm_key_size=32, comm_value_size=100, comm_query_size=32,
				 num_comm_heads=4, comm_dropout=0.1,
				 UnitActivityMask=None,Number_active=3):
		super().__init__()
		if comm_value_size != hidden_size:
			# print('INFO: Changing communication value size to match hidden_size')
			comm_value_size = hidden_size
		self.device = device
		self.hidden_size = hidden_size
		self.num_units = num_units
		self.rnn_cell = rnn_cell
		self.key_size = input_key_size
		self.k = k
		self.num_input_heads = num_input_heads
		self.num_comm_heads = num_comm_heads
		self.input_key_size = input_key_size
		self.input_query_size = input_query_size
		self.input_value_size = input_value_size

		self.comm_key_size = comm_key_size
		self.comm_query_size = comm_query_size
		self.comm_value_size = comm_value_size

		self.key = nn.Linear(input_size, num_input_heads * input_query_size).to(self.device)
		self.value = nn.Linear(input_size, num_input_heads * input_value_size).to(self.device)

		#########number of schemas and schema weightings#######
		self.schema_weighting = schema_weighting
		self.num_schemas = num_schemas
		self.Number_active = Number_active

		if self.rnn_cell == 'GRU':
			self.rnn = GroupGRUCell_sharedParameters(self.device,input_value_size, hidden_size, num_units,schema_weighting,num_schemas,self.Number_active)
			#self.rnn = GroupGRUCell(input_value_size, hidden_size, num_units)
			self.query = GroupLinearLayer(hidden_size, input_key_size * num_input_heads, self.num_units)
		else:
			self.rnn = GroupLSTMCell_sharedParameters(self.device,input_value_size, hidden_size, num_units,schema_weighting,num_schemas,self.Number_active)
			self.query = GroupLinearLayer(hidden_size, input_key_size * num_input_heads, self.num_units)
		self.query_ = GroupLinearLayer(hidden_size, comm_query_size * num_comm_heads, self.num_units)
		self.key_ = GroupLinearLayer(hidden_size, comm_key_size * num_comm_heads, self.num_units)
		self.value_ = GroupLinearLayer(hidden_size, comm_value_size * num_comm_heads, self.num_units)
		self.comm_attention_output = GroupLinearLayer(num_comm_heads * comm_value_size, comm_value_size, self.num_units)
		self.comm_dropout = nn.Dropout(p=input_dropout)
		self.input_dropout = nn.Dropout(p=comm_dropout)

		##########unity activation mask, 1 means active 0 mean inactive#######
		if UnitActivityMask is None:
			print ("UnitActivityMask not in use ")
			self.UnitActivityMask=np.repeat(1,self.num_units )
		else:
			self.UnitActivityMask =UnitActivityMask

			print("using UnitActivityMask")
			print(self.UnitActivityMask)
		self.UnitActivityMask =torch.from_numpy(self.UnitActivityMask).to(self.device)




	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
		new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def input_attention_mask(self, x, h):
		"""
        Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
                h (batch_size, num_units, hidden_size)
        Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
                mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
        """
		key_layer = self.key(x)
		value_layer = self.value(x)
		query_layer = self.query(h)

		key_layer = self.transpose_for_scores(key_layer, self.num_input_heads, self.input_key_size)
		value_layer = torch.mean(self.transpose_for_scores(value_layer, self.num_input_heads, self.input_value_size),
								 dim=1)
		query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size)
		attention_scores = torch.mean(attention_scores, dim=1)

		#######setting attention score zeros for inactive units######
		#attention_scores=attention_scores*self.UnitActivityMask
		#print("input attention scores:")
		#print(attention_scores)

		mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)

		not_null_scores = attention_scores[:, :, 0]
		topk1 = torch.topk(not_null_scores, self.k, dim=1)
		row_index = np.arange(x.size(0))
		row_index = np.repeat(row_index, self.k)

		mask_[row_index, topk1.indices.view(-1)] = 1

		attention_probs = self.input_dropout(nn.Softmax(dim=-1)(attention_scores))
		inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)

		return inputs, mask_

	def communication_attention(self, h, mask):
		"""
        Input : h (batch_size, num_units, hidden_size)
                mask obtained from the input_attention_mask() function
        Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
        """
		query_layer = []
		key_layer = []
		value_layer = []

		query_layer = self.query_(h)
		key_layer = self.key_(h)
		value_layer = self.value_(h)

		query_layer = self.transpose_for_scores(query_layer, self.num_comm_heads, self.comm_query_size)
		key_layer = self.transpose_for_scores(key_layer, self.num_comm_heads, self.comm_key_size)
		value_layer = self.transpose_for_scores(value_layer, self.num_comm_heads, self.comm_value_size)
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.comm_key_size)

		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		mask = [mask for _ in range(attention_probs.size(1))]
		mask = torch.stack(mask, dim=1)

		attention_probs = attention_probs * mask.unsqueeze(3)
		attention_probs = self.comm_dropout(attention_probs)
		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		context_layer = self.comm_attention_output(context_layer)
		context_layer = context_layer + h

		return context_layer

	def forward(self, x, hs, cs=None):
		"""
		Input : x (batch_size, 1 , input_size)
				hs (batch_size, num_units, hidden_size)
				cs (batch_size, num_units, hidden_size)
		Output: new hs, cs for LSTM
				new hs for GRU
		"""
		size = x.size()
		null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
		x = torch.cat((x, null_input), dim=1)

		# Compute input attention
		inputs, mask = self.input_attention_mask(x, hs)
		h_old = hs * 1.0
		if cs is not None:
			c_old = cs * 1.0

		# Compute RNN(LSTM or GRU) output

		if cs is not None:
			hs, cs = self.rnn(inputs, (hs, cs))
		else:
			hs = self.rnn(inputs, hs)

		# Block gradient through inactive units
		mask = mask.unsqueeze(2)
		h_new = blocked_grad.apply(hs, mask)

		# Compute communication attention
		h_new = self.communication_attention(h_new, mask.squeeze(2))

		hs = mask * h_new + (1 - mask) * h_old
		if cs is not None:
			cs = mask * cs + (1 - mask) * c_old
			return hs, cs

		return hs, None



class RIM(nn.Module):
	def __init__(self, device, input_size, hidden_size, num_units, k, rnn_cell, n_layers, bidirectional, **kwargs):
		super().__init__()
		if device == 'cuda':
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.n_layers = n_layers
		self.num_directions = 2 if bidirectional else 1
		self.rnn_cell = rnn_cell
		self.num_units = num_units
		self.hidden_size = hidden_size
		if self.num_directions == 2:
			self.rimcell = nn.ModuleList([RIMCell(self.device, input_size, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) if i < 2 else
				RIMCell(self.device, 2 * hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) for i in range(self.n_layers * self.num_directions)])
		else:
			self.rimcell = nn.ModuleList([RIMCell(self.device, input_size, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) if i == 0 else
			RIMCell(self.device, hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) for i in range(self.n_layers)])

	def layer(self, rim_layer, x, h, c = None, direction = 0):
		batch_size = x.size(1)
		xs = list(torch.split(x, 1, dim = 0))
		if direction == 1: xs.reverse()
		hs = h.squeeze(0).view(batch_size, self.num_units, -1)
		cs = None
		if c is not None:
			cs = c.squeeze(0).view(batch_size, self.num_units, -1)
		outputs = []
		for x in xs:
			x = x.squeeze(0)
			hs, cs = rim_layer(x.unsqueeze(1), hs, cs)
			outputs.append(hs.view(1, batch_size, -1))
		if direction == 1: outputs.reverse()
		outputs = torch.cat(outputs, dim = 0)
		if c is not None:
			return outputs, hs.view(batch_size, -1), cs.view(batch_size, -1)
		else:
			return outputs, hs.view(batch_size, -1)

	def forward(self, x, h = None, c = None):
		"""
		Input: x (seq_len, batch_size, feature_size
			   h (num_layers * num_directions, batch_size, hidden_size * num_units)
			   c (num_layers * num_directions, batch_size, hidden_size * num_units)
		Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
				h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
		"""

		hs = torch.split(h, 1, 0) if h is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
		hs = list(hs)
		cs = None
		if self.rnn_cell == 'LSTM':
			cs = torch.split(c, 1, 0) if c is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
			cs = list(cs)
		for n in range(self.n_layers):
			idx = n * self.num_directions
			if cs is not None:
				x_fw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx])
			else:
				x_fw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None)
			if self.num_directions == 2:
				idx = n * self.num_directions + 1
				if cs is not None:
					x_bw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx], direction = 1)
				else:
					x_bw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None, direction = 1)

				x = torch.cat((x_fw, x_bw), dim = 2)
			else:
				x = x_fw
		hs = torch.stack(hs, dim = 0)
		if cs is not None:
			cs = torch.stack(cs, dim = 0)
			return x, hs, cs
		return x, hs



class RIM_SharedParameters(nn.Module):
	def __init__(self, device, input_size, hidden_size, num_units, k, rnn_cell, n_layers, bidirectional,UnitActivityMask = None,Number_active = 3,**kwargs):
		super().__init__()
		if device == 'cuda':
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.n_layers = n_layers
		self.num_directions = 2 if bidirectional else 1
		self.rnn_cell = rnn_cell
		self.num_units = num_units
		self.hidden_size = hidden_size

		self.UnitActivityMask = UnitActivityMask
		self.Number_active = Number_active

		if self.num_directions == 2:
			self.rimcell = nn.ModuleList([RIMCell_SharedParameters(self.device, input_size, hidden_size, num_units, k, rnn_cell,UnitActivityMask,Number_active).to(self.device) if i < 2 else
				RIMCell_SharedParameters(self.device, 2 * hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, UnitActivityMask,Number_active).to(self.device) for i in range(self.n_layers * self.num_directions)])
		else:
			self.rimcell = nn.ModuleList([RIMCell_SharedParameters(self.device, input_size, hidden_size, num_units, k, rnn_cell, UnitActivityMask,Number_active).to(self.device) if i == 0 else
			RIMCell_SharedParameters(self.device, hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, UnitActivityMask,Number_active).to(self.device) for i in range(self.n_layers)])

	def layer(self, rim_layer, x, h, c = None, direction = 0):
		batch_size = x.size(1)
		xs = list(torch.split(x, 1, dim = 0))
		if direction == 1: xs.reverse()
		hs = h.squeeze(0).view(batch_size, self.num_units, -1)
		cs = None
		if c is not None:
			cs = c.squeeze(0).view(batch_size, self.num_units, -1)
		outputs = []
		for x in xs:
			x = x.squeeze(0)
			hs, cs = rim_layer(x.unsqueeze(1), hs, cs)
			outputs.append(hs.view(1, batch_size, -1))
		if direction == 1: outputs.reverse()
		outputs = torch.cat(outputs, dim = 0)
		if c is not None:
			return outputs, hs.view(batch_size, -1), cs.view(batch_size, -1)
		else:
			return outputs, hs.view(batch_size, -1)

	def forward(self, x, h = None, c = None):
		"""
		Input: x (seq_len, batch_size, feature_size
			   h (num_layers * num_directions, batch_size, hidden_size * num_units)
			   c (num_layers * num_directions, batch_size, hidden_size * num_units)
		Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
				h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
		"""

		hs = torch.split(h, 1, 0) if h is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
		hs = list(hs)
		cs = None
		if self.rnn_cell == 'LSTM':
			cs = torch.split(c, 1, 0) if c is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
			cs = list(cs)
		for n in range(self.n_layers):
			idx = n * self.num_directions
			if cs is not None:
				x_fw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx])
			else:
				x_fw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None)
			if self.num_directions == 2:
				idx = n * self.num_directions + 1
				if cs is not None:
					x_bw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx], direction = 1)
				else:
					x_bw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None, direction = 1)

				x = torch.cat((x_fw, x_bw), dim = 2)
			else:
				x = x_fw
		hs = torch.stack(hs, dim = 0)
		if cs is not None:
			cs = torch.stack(cs, dim = 0)
			return x, hs, cs
		return x, hs


if __name__ == "__main__":
	class Arguments:
		cuda=False
		epochs=3
		train_size=555
		test_size=333

		batch_size=500
		hidden_size=100
		input_size=1
		T=50

		InputRange=range(1,8)
		InputLength=8
		Gap=0
		indicator=9
		Gaplength=6

		num_units=9
		rnn_cell="LSTM"
		key_size_input=128
		value_size_input=400
		query_size_input=128
		num_input_heads=1
		num_comm_heads=4
		input_dropout=0.1
		comm_dropout=0.1

		Number_active=3

		key_size_comm=128
		value_size_comm=100
		query_size_comm=128
		k=3

		size=14
		loadsaved=0
		initial_model=None

		log_dir='copying_task_SharedParameters'
		UnitActivityMask=None

		num_schemas=20
		schema_weighting = torch.randn((num_units, num_schemas))


	# RIM cell
	args=vars(Arguments)
	device="cpu"
	rim_model =  RIMCell_SharedParameters(device,args['input_size'], args['hidden_size'], args['num_units'], args['k'],
								 args['rnn_cell'],
								 args['schema_weighting'], args['num_schemas'],
								 args['key_size_input'], args['value_size_input'],
								 args['query_size_input'],
								 args['num_input_heads'], args['input_dropout'], args['key_size_comm'],
								 args['value_size_comm'], args['query_size_comm'], args['num_input_heads'],
								 args['comm_dropout'],
								args['UnitActivityMask'],args['Number_active']).to(device)

	BatchSize=args["batch_size"]
	num_units=args['num_units']
	hiddenSize=args['hidden_size']
	din=args['input_size']

	Seq_length=2*args["InputLength"]+args["Gaplength"]
	x = torch.randn(BatchSize, Seq_length, din)
	#y = (torch.randn(BatchSize, num_units, hiddenSize),
	#	 torch.randn(BatchSize, num_units, hiddenSize))
	y = torch.randn(BatchSize, num_units, hiddenSize)

	#y = torch.randn(BatchSize, 9)

	x = x.float()
	hs = torch.randn(x.size(0), args['num_units'], args['hidden_size']).to(device)
	cs = None
	if args['rnn_cell'] == 'LSTM':
		cs = torch.randn(x.size(0), args['num_units'], args['hidden_size']).to(device)


	xs = torch.split(x, 1, 1)
	preds_ = []
	loss = 0
	loss_last_10 = 0

	Loss = nn.MSELoss()
	#Linear = nn.Linear(args['hidden_size'] * args['num_units'], 9)

	optimizer = torch.optim.Adam(rim_model.parameters(), lr=0.0005)
	optimizer.zero_grad()


	for i, k in enumerate(xs):
		hs, cs = rim_model(k, hs, cs)


		#preds = Linear(hs.contiguous().view(x.size(0), -1))
		preds=hs
		preds_.append(preds)
		if y is not None:
			loss += Loss(preds, y)
			if i >= len(xs) - 10:
				loss_last_10 += Loss(preds, y)
	preds_ = torch.stack(preds_, dim=1)

	loss.backward(retain_graph=True)
	optimizer.step()



	#whole RIM module
	args["input_size"]
	args.keys()

	RIM_model=RIM(device="cpu", input_size=args["input_size"],
								   hidden_size=args["hidden_size"], num_units=args["num_units"],
								   k=args["k"], rnn_cell=args["rnn_cell"],
								   n_layers=1, bidirectional=False)

	x = torch.randn(Seq_length,BatchSize, din)
	# y = (torch.randn(BatchSize, num_units, hiddenSize),
	#	 torch.randn(BatchSize, num_units, hiddenSize))
	y = torch.randn(BatchSize, num_units, hiddenSize)

	x = x.float()
	hs = torch.randn(1,BatchSize,num_units*hiddenSize).to(device)
	cs = None
	if args['rnn_cell'] == 'LSTM':
		cs = torch.randn(1,BatchSize,num_units*hiddenSize).to(device)

	Output, hs, cs = RIM_model(x, hs, cs)
	x.shape
	hs.shape
	cs.shape


	Loss = nn.MSELoss()
	# Linear = nn.Linear(args['hidden_size'] * args['num_units'], 9)

	optimizer = torch.optim.Adam(rim_model.parameters(), lr=0.0005)
	
	optimizer.zero_grad()

	Output, hs, cs = RIM_model(x, hs, cs)

	y = torch.randn(Output.shape)

	loss=Loss(y,Output)



	# whole RIM module with shared parameters
	args["input_size"]
	args.keys()

	RIM_model = RIM_SharedParameters(device="cpu", input_size=args["input_size"],
					hidden_size=args["hidden_size"], num_units=args["num_units"],
					k=args["k"], rnn_cell=args["rnn_cell"],
					n_layers=1, bidirectional=False,
					UnitActivityMask = None,Number_active = 3)

	#RIM_mmodel=torch.nn.LSTM(input_size=args["input_size"],hidden_size=num_units * hiddenSize)
	x = torch.randn(Seq_length, BatchSize, din)
	# y = (torch.randn(BatchSize, num_units, hiddenSize),
	#	 torch.randn(BatchSize, num_units, hiddenSize))
	y = torch.randn(BatchSize, num_units, hiddenSize)

	x = x.float()
	hs = torch.randn(1, BatchSize, num_units * hiddenSize).to(device)
	cs = None
	if args['rnn_cell'] == 'LSTM':
		cs = torch.randn(1, BatchSize, num_units * hiddenSize).to(device)

	Output, hs, cs = RIM_model(x, hs, cs)
	x.shape
	hs.shape
	cs.shape

	Loss = nn.MSELoss()
	# Linear = nn.Linear(args['hidden_size'] * args['num_units'], 9)

	optimizer = torch.optim.Adam(rim_model.parameters(), lr=0.0005)

	optimizer.zero_grad()

	Output, hs, cs = RIM_model(x, hs, cs)

	y = torch.randn(Output.shape)

	loss = Loss(y, Output)