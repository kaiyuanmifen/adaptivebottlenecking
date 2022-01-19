import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from make_env import make_env
import argparse

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--Method', type=str, default="Baseline",
                    help='which method to use')

    parser.add_argument('--Round', type=int, default=0,
        help='random seed round')


    args = parser.parse_args()

    print("Method:")
    print(args.Method)
    print("Rounds",args.Round)

    #scenario = 'simple'
    scenario = 'simple_adversary'
    env = make_env(scenario)
    n_agents = env.n
    print("n_agents")
    print(n_agents)
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    

    critic_dims = sum(actor_dims)

    Save_DIR='./tmpmodel/maddpg_'+args.Method+"_"+str(args.Round)+"/"
    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space[0].n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir=Save_DIR,args=args)

    pytorch_total_params = sum(p.numel() for p in maddpg_agents.agents[1].actor.parameters())

    print("number of MADDPG actor parameters:", pytorch_total_params)
    

    import os
    if not os.path.exists(Save_DIR):
        os.makedirs(Save_DIR)

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 100
    N_GAMES = 50000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video

            
            actions,All_CBLoss,All_att_scores = maddpg_agents.choose_action(obs)


            obs_, reward, done, info = env.step(actions)

            # print("obs")
            # print([obs[i].shape for i in range(len(obs))])


            # print("obs_")
            # print([obs_[i].shape for i in range(len(obs_))])


            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)


            # print("state")
            # print(state.shape)

            # print("state_ ")
            # print(state_ .shape)


            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])


        import csv   
        fields=[args.Method,i,avg_score,All_CBLoss,All_att_scores]

        with open(Save_DIR+'/Results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
