import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
#from torch.utils.tensorboard import SummaryWriter
#import gym_lunar_lander_custom
import argparse
import torch


from QuantizerFunction import QuantizerFunction

n_actions = 4
input_dims = 24

#writer = SummaryWriter()



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = F.relu(self.fc_2(x))
        return x






class DeepQNetwork_Q(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions,args):
        super(DeepQNetwork_Q, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.QuantizerFunction=QuantizerFunction(*self.input_dims,CodebookSize=16,Method=args.Method)


        self.args=args
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state=state.unsqueeze(1)
        state,CBloss,att_scores=self.QuantizerFunction(state)
        state=state.squeeze(1)


        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions,CBloss,att_scores




class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,args,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork_Q(lr, n_actions=n_actions, input_dims=input_dims,
                                  fc1_dims=256, fc2_dims=256,args=args)

      

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions,ExtraLoss,_ = self.Q_eval.forward(state)

            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
            ExtraLoss=0


        return action,ExtraLoss

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval,ExtraLoss,_ = self.Q_eval.forward(state_batch)
        q_eval=q_eval[batch_index, action_batch]
      


        q_next,ExtraLoss_next,_ = self.Q_eval.forward(new_state_batch)
        
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        beta=1
        loss=loss+beta*(ExtraLoss+ExtraLoss_next)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min



def train(env,agent):
        agent.Q_eval.train()
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action,ExtraLoss = agent.choose_action(observation)
           
            observation_, reward, done, info = env.step(action)


            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_


        return score,agent.epsilon,ExtraLoss





def test(env,agent):
        agent.Q_eval.eval()
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action,ExtraLoss = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)


            score += reward

            observation = observation_


        return score,ExtraLoss



if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--Round', type=int, default=None,
        help='Round of Random seeds')

    parser.add_argument('--data', type=str, default='CartPole-v1',
        help='which data to use')


    parser.add_argument('--Method', type=str, default="Original",
    help='method name')


    args = parser.parse_args()



    env = gym.make(args.data)

    observation = env.reset()
    ObservationSpace=observation.shape[0]
    print("ObservationSpace")
    print(ObservationSpace)
    action_space = env.action_space.n
    print("action_space")
    print(action_space)


    Saving_Interval=10

    # if args.data=="LunarLander-v2":
        
    #     env_testOOD = gym.make('LunarLanderOOD-v0')    
    # else:

    env_testOOD = gym.make(args.data)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=action_space, eps_end=0.01,
                  input_dims=[ObservationSpace], lr=0.001,args=args)
    scores, eps_history = [], []
    n_games = 2000
    test_scores,testOOD_scores=[],[]
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        score,epsilon,ExtraLoss=train(env,agent)

        scores.append(score)
        eps_history.append(epsilon)

        avg_score = np.mean(scores[-100:])
        #writer.add_scalar('average score', avg_score, i)
        #T.save(agent.Q_eval.state_dict(), 'checkpoint_policy.pth')


        if i%Saving_Interval==0:

            ####test on original env
            test_score,test_ExtraLoss=test(env,agent)
            test_scores.append(test_score)
            test_avg_score = np.mean(test_scores[-100:])

            ####test on OOD env
            testOOD_score,testOOD_ExtraLoss=test(env_testOOD,agent)
            testOOD_scores.append(testOOD_score)
            testOOD_avg_score = np.mean(testOOD_scores[-100:])



            import csv   
            fields=[args.data,args.Method,i,score,avg_score,epsilon,float(ExtraLoss),test_score,test_avg_score,float(test_ExtraLoss),testOOD_score,testOOD_avg_score,float(testOOD_ExtraLoss)]
            print("fileds")
            print(fields)
            with open(r'../../_DeepQResults.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)



            print('episode ', i, 'score %.2f' % score,
                  'average score %.2f' % avg_score,
                  'epsilon %.2f' % agent.epsilon)