from env_Drones import EnvDrones
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np
env = EnvDrones(50, 4, 10, 30, 6)   # map_size, drone_num, view_range, tree_num, human_num
env.rand_reset_drone_pos()
max_MC_iter = 10

#plt.ion()
#fig = plt.figure()

#gs = GridSpec(1, 2, figure=fig)
#ax1 = fig.add_subplot(gs[0:1, 0:1])
#ax2 = fig.add_subplot(gs[0:1, 1:2])
for MC_iter in range(max_MC_iter):
    print(MC_iter)


    Full_obs=env.get_full_obs()
    Full_obs=Full_obs.astype(np.uint8)
    im = Image.fromarray(Full_obs)
    im.save("Full_obs.png")


    joint_obs=env.get_joint_obs()
    # print("obj observed")
    #print(joint_obs[:,:,0])
    # print((np.sum(joint_obs[:,:,0]==1.0)))
    #print(joint_obs.shape)

    #joint_obs[:,:,1][joint_obs[:,:,1]==1]=0
    reward=np.sum((joint_obs[:,:,0]==1)*(joint_obs[:,:,1]==0)*(joint_obs[:,:,2]==0))
    # print("human view")
    # print(joint_obs.shape)
    # print(joint_obs[:,:,0])
    # print(joint_obs[:,:,0].shape)
    # print(np.unique(joint_obs))
    
    print("reward")
    print(reward)



    joint_obs=(joint_obs*255).astype(np.uint8)
    
    im = Image.fromarray(joint_obs)
    im.save("joint_obs.png")


    full_obs=env.get_full_obs()
    # print("full view")
    # print(np.sum((full_obs[:,:,0]==1)*(full_obs[:,:,1]==0)*(full_obs[:,:,2]==0)))

    full_obs=(full_obs*255).astype(np.uint8)

    im = Image.fromarray(full_obs)
    im.save("full_obs.png")

    # print("Number of drones")

    # print(len(env.drone_list))

    for indx in range(len(env.drone_list)):
        Obs_drone=env.get_drone_obs(env.drone_list[indx])
        Obs_drone=Obs_drone.astype(np.uint8)
        im = Image.fromarray(255*Obs_drone)
        im.save("agent"+str(indx)+"_obs.jpeg")
   #ax1.imshow(env.get_full_obs())
    #ax2.imshow(env.get_joint_obs())

    human_act_list = []
    for i in range(env.human_num):
        human_act_list.append(random.randint(0, 4))

    drone_act_list = []
    for i in range(env.drone_num):
        drone_act_list.append(random.randint(0, 4))
    env.step(human_act_list, drone_act_list)
    #plt.pause(.5)
    #plt.draw()
#plt.close(fig)
#plt.savefig("testingFigure.png")