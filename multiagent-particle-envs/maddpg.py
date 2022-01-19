import torch as T
import torch.nn.functional as F
from agent import Agent
import torch

from QuantizerFunction import QuantizerFunction

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/',args=None):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario


        self.args=args
        self.QuantizerFunctions=[]


        for agent_idx in range(self.n_agents):

            self.QuantizerFunctions.append(QuantizerFunction(actor_dims[agent_idx], name="Quantizer_"+str(agent_idx), chkpt_dir=chkpt_dir,args=args))

            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir,args=args))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent_idx in range(self.n_agents):
            self.agents[agent_idx].save_models()
            self.QuantizerFunctions[agent_idx].save_models()
   
        
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent_idx in range(self.n_agents):
            self.agents[agent_idx].load_models()
            self.QuantizerFunctions[agent_idx].load_models()


    def choose_action(self, raw_obs):
        actions = []
        All_CBloss=0
        All_att_scores=[]

        for agent_idx, agent in enumerate(self.agents):
            action,CBloss,att_scores = agent.choose_action(raw_obs[agent_idx],self.QuantizerFunctions[agent_idx])
            actions.append(action)
            All_CBloss=All_CBloss+CBloss
            All_att_scores.append(att_scores.unsqueeze(0))

        All_att_scores=T.cat(All_att_scores,0)

        return actions,All_CBloss,All_att_scores

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()
   
        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        all_CBLoss=[]
        
        ####quantize each of the actor states

        for agent_idx, agent in enumerate(self.agents):

            actor_new_states[agent_idx],new_CBloss,new_att_scores = self.QuantizerFunctions[agent_idx](T.tensor(actor_new_states[agent_idx], 
                         dtype=T.float).to(device))

            actor_states[agent_idx],CBloss,att_scores = self.QuantizerFunctions[agent_idx](T.tensor(actor_states[agent_idx], 
                             dtype=T.float).to(device))


            all_CBLoss.append(new_CBloss+CBloss)




        #####actors

        for agent_idx, agent in enumerate(self.agents):
            new_states = actor_new_states[agent_idx]

            new_pi = agent.target_actor.forward(new_states)
        
            all_agents_new_actions.append(new_pi)


            mu_states =actor_states[agent_idx]


            pi= agent.actor.forward(mu_states)
            

            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        #####regenerate the states and states_ from actor state after discretization
  
        states=torch.cat(actor_states,1)
        states_=torch.cat(actor_new_states,1)



        #get values 

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            if self.args.Method!="Original":
                self.QuantizerFunctions[agent_idx].optimizer.zero_grad()
                CodebookLoss=all_CBLoss[agent_idx]
                CodebookLoss.backward(retain_graph=True)
                self.QuantizerFunctions[agent_idx].optimizer.step()
               # print("updating CodebookLoss")


            agent.update_network_parameters()
