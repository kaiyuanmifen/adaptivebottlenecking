import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from RIM import RIMCell_SharedParameters,RIMCell
from QuantizerFunction import QuantizerFunction
#NUM_UNITS = 4

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        try:
            m.weight.data.normal_(0, 1)
            m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
            if m.bias is not None:
                m.bias.data.fill_(0)
        except:
            pass

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self,args, obs_space, action_space, use_memory=True,
                 use_text=False, use_rim = False,NUM_UNITS=4,
                 k=3,rnn_cell="LSTM",UnitActivityMask=None, Number_active=3,device="cpu",
                 schema_weighting=None, num_schemas=8,rim_type="Original"):
        super().__init__()
        """
        NUM_UNITS needs to be compatible with inpu_size , under current setting,
        2,4,8... are ok
        """
        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.use_rim = use_rim

        self.NUM_UNITS=NUM_UNITS

        print("Number of units: " + str(NUM_UNITS))

        if bool(use_rim):
            print("k: " + str(k))
            print('Number_active: ' + str(Number_active))
            print("num_schemas: " + str(num_schemas))

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        

        # Define memory
        if self.use_memory:
            if use_rim:
                if rim_type=="Shared":
                    print("using RIM_sharedParameters")

                    if schema_weighting is None:
                        schema_weighting = [torch.randn((NUM_UNITS, num_schemas)),torch.randn((NUM_UNITS, num_schemas))]#the current model has two schema weight

                    self.memory_rnn = RIMCell_SharedParameters(device=device,
                                                               input_size=self.image_embedding_size,
                                                               hidden_size=self.semi_memory_size // NUM_UNITS,
                                                               num_units=NUM_UNITS, k=k, rnn_cell=rnn_cell,
                                                               input_value_size=64,
                                                               comm_value_size=self.semi_memory_size // NUM_UNITS,
                                                               UnitActivityMask=UnitActivityMask, Number_active=Number_active,
                                                               schema_weighting=schema_weighting, num_schemas=num_schemas)

                elif rim_type=="Original":
                    print("using original RIM")
                    self.memory_rnn = RIMCell(device, self.image_embedding_size,
                                              self.semi_memory_size // NUM_UNITS, NUM_UNITS, k, 'LSTM', input_value_size=64,
                                              comm_value_size=self.semi_memory_size // NUM_UNITS)

                else:
                    raise Exception("please use valid RIM types")

            else:
                print("Not using RIM")
                self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        #####quantization
        self.args=args
        self.QuantizerFunction=QuantizerFunction(64,CodebookSize=96,args=args,N_factors=[1,2,4])
        # Initialize parameters correctly
        self.apply(init_params)






        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        x,ExtraLoss,att_scores=self.QuantizerFunction(x)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            if self.use_rim:    
                hidden = list(hidden)
                hidden[0] = hidden[0].view(hidden[0].size(0), self.NUM_UNITS, -1)
                hidden[1] = hidden[0].view(hidden[1].size(0), self.NUM_UNITS, -1)
                x = x.unsqueeze(1)

                hidden = self.memory_rnn(x, hidden[0], hidden[1])
                hidden = list(hidden)
                hidden[0] = hidden[0].view(hidden[0].size(0), -1)
                hidden[1] = hidden[1].view(hidden[1].size(0), -1)
            else:
                hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory,ExtraLoss

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


if __name__ == "__main__":
    import utils
    args_procs=16
    args_seed=1
    args_env="MiniGrid-RedBlueDoors-6x6-v0"
    envs = []
    for i in range(args_procs):
        envs.append(utils.make_env(args_env, args_seed + 10000 * i))

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    acmodel=ACModel(obs_space, envs[0].action_space)
    acmodel.parameters()

    for p in acmodel.parameters():
        print (p.grad)
    sum(p.grad.data.norm(2).item() ** 2 for p in acmodel.parameters() if p.grad is not None) ** 0.5