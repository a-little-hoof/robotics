import numpy as np

def create_robot(p0, v, theta, sim_time, num_timesteps):
    ### p0 starting point
    ### theta direction
    t = np.linspace(0, sim_time, num_timesteps) #0-5 50等分
    # print(t)
    # print(t.shape)
    theta = theta * np.ones(np.shape(t))
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    v = np.stack([vx, vy])
    # print(v)
    # print(v.shape)
    p0 = p0.reshape((2, 1))
    p = p0 + np.cumsum(v, axis=1) * (sim_time/num_timesteps) 
    # print(p.shape)
    #速度逐渐累加 绘制用
    p_state_history = np.concatenate((p, v))
    # print(p)
    # print(p.shape) (4, 50)
    # print(1)
    return p_state_history
    

def create_obstacles(sim_time, num_timesteps, speed, start_obs, directions, obj_num):
    obstacles = None
    for i in range(obj_num):
        v = speed[i]
        p0 = np.array(start_obs[i])
        direction = directions[i]
        obst = create_robot(p0, v, direction, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
        if i==0:
            obstacles = obst
        else:
            obstacles = np.dstack((obstacles, obst))

    return obstacles
