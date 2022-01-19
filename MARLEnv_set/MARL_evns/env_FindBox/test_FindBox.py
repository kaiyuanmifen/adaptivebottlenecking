from env_FindBox import EnvFindBox as Env
import random
import numpy as np
from PIL import Image

if __name__ == '__main__':
    env = Env(15)
    max_iter = 100000
    for i in range(max_iter):
        #print("iter= ", i)
        #env.plot_scene()
        action_list = [random.randint(0, 3), random.randint(0, 3)]
        reward, done = env.step(action_list)
        if done:
            print('find goal, reward', reward,"iter",str(i))

            joint_obs=env.get_global_obs()
            joint_obs=(joint_obs*255).astype(np.uint8)
            im = Image.fromarray(joint_obs)
            im.save("Found"+".png")


            agent_obs=env.get_agt1_obs()
            agent_obs=(agent_obs*255).astype(np.uint8)
            im = Image.fromarray(agent_obs)
            im.save("Found_agent1"+".png")


            agent_obs=env.get_agt2_obs()
            agent_obs=(agent_obs*255).astype(np.uint8)
            im = Image.fromarray(agent_obs)
            im.save("Found_agent2"+".png")



            env.reset()

            break