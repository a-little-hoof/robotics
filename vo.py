"""
Local search using Velocity-obstacle method
假设障碍物速度恒定
障碍物起始点，速度可自定义

"""

from utils import create_obstacles
from draw import plot_robot_and_obstacles
import numpy as np
import random 

### configurations
SIM_TIME = 5
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
# robot config
ROBOT_NUM = 3
ROBOT_RADIUS = 0.5
VMAX = 2
VMIN = 0.2
START_POINT = [[5, 5, 0, 0], [2, 3, 0, 0],[1,10,0,0],[10,1,0,0]]#, [8, 9, 0, 0]]
GOAL = [[5, 5, 0, 0], [8, 8, 0, 0],[10,1,0,0],[1,10,0,0]]#, [9, 8, 0, 0]]
# obstacle config
OBJ_NUM = 4
SPEED = [-2, 2, 2, 2]
START_OBS = [[5, 7],[3, 5],[7, 7],[7.5, 2.5]]
DIRECIONS = [np.pi/2,0,-np.pi*3/4,np.pi*3/4]

THRES = 3

def simulate(filename):
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS, SPEED, START_OBS, DIRECIONS, OBJ_NUM)
    robot_state_histories = []
    for j in range(ROBOT_NUM):
        start = np.array(START_POINT[j])
        goal = np.array(GOAL[j])

        pos = start
        robot_state_history = np.empty((4, NUMBER_OF_TIMESTEPS))
        for i in range(NUMBER_OF_TIMESTEPS):
            v = compute_desired_velocity(pos, goal)

            control_vel = compute_velocity(pos, obstacles[:, i, :], v, i, j, robot_state_histories)
            # update pos
            new_pos = np.empty((4))
            new_pos[:2] = pos[:2] + control_vel * TIMESTEP
            new_pos[-2:] = control_vel
            pos = new_pos
            robot_state_history[:, i] = pos
        robot_state_histories.append(robot_state_history)

    plot_robot_and_obstacles(robot_state_histories, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)

def compute_desired_velocity(current_pos, goal_pos):
    disp_vec = (goal_pos - current_pos)[:2]
    # print(current_pos)
    # print(disp_vec)
    norm = np.linalg.norm(disp_vec)
    if norm < ROBOT_RADIUS/5:
        return np.zeros(2)
    disp_vec = disp_vec / norm
    # 单位向量
    desired_vel = VMAX * disp_vec
    return desired_vel

def compute_velocity(pos, obstacles, v_desired, timestep, robot_ind, robot_state_histories):
    thres = 1
    pA = pos[:2]
    vA = pos[-2:]
    # Compute the constraints
    # for each velocity obstacles
    number_of_obstacles = obstacles.shape[1]
    A = np.empty((number_of_obstacles * 2 + 2*robot_ind, 2)) # matrix
    b = np.empty((number_of_obstacles * 2 + 2*robot_ind)) # vector
    for i in range(number_of_obstacles):
        obstacle = obstacles[:, i]
        # print(obstacle.shape)
        pB = obstacle[:2]
        vB = obstacle[2:]
        dispBA = pA - pB #圆心和偏转角
        distBA = np.linalg.norm(dispBA)
        if distBA < THRES:
            thres = 0
        thetaBA = np.arctan2(dispBA[1], dispBA[0])
        if 2.2 * ROBOT_RADIUS > distBA:
            distBA = 2.2*ROBOT_RADIUS
        phi_obst = np.arcsin(2.2*ROBOT_RADIUS/distBA)
        phi_left = thetaBA + phi_obst
        phi_right = thetaBA - phi_obst #确定左右两条切线

        # VO
        translation = vB
        Atemp, btemp = create_constraints(translation, phi_left, "left")
        A[i*2, :] = Atemp
        b[i*2] = btemp
        Atemp, btemp = create_constraints(translation, phi_right, "right")
        A[i*2 + 1, :] = Atemp
        b[i*2 + 1] = btemp

    for i in range(robot_ind):
        obstacle = robot_state_histories[i][:,timestep]
        # print(robot_state_histories[i].shape)
        # print(obstacle)
        pB = obstacle[:2]
        vB = obstacle[2:]
        dispBA = pA - pB #圆心和偏转角
        distBA = np.linalg.norm(dispBA)
        if distBA < THRES:
            thres = 0
        thetaBA = np.arctan2(dispBA[1], dispBA[0])
        if 2.2 * ROBOT_RADIUS > distBA:
            distBA = 2.2*ROBOT_RADIUS
        phi_obst = np.arcsin(2.2*ROBOT_RADIUS/distBA)
        phi_left = thetaBA + phi_obst
        phi_right = thetaBA - phi_obst #确定左右两条切线

        # VO
        translation = vB
        Atemp, btemp = create_constraints(translation, phi_left, "left")
        A[i*2 + 2*number_of_obstacles, :] = Atemp
        b[i*2 + 2*number_of_obstacles] = btemp
        Atemp, btemp = create_constraints(translation, phi_right, "right")
        A[i*2 + 1 + 2*number_of_obstacles, :] = Atemp
        b[i*2 + 1 + 2*number_of_obstacles] = btemp
    # Create search-space
    th = np.linspace(0, 2*np.pi, 20)
    vel = np.linspace(0, VMAX, 5)

    vv, thth = np.meshgrid(vel, th)

    vx_sample = (vv * np.cos(thth)).flatten()
    vy_sample = (vv * np.sin(thth)).flatten()

    v_sample = np.stack((vx_sample, vy_sample))

    v_satisfying_constraints = check_constraints(v_sample, A, b)
    # Objective function
    size = np.shape(v_satisfying_constraints)[1]
    diffs = v_satisfying_constraints - \
        ((v_desired).reshape(2, 1) @ np.ones(size).reshape(1, size))
    norm = np.linalg.norm(diffs, axis=0)
    min_index = np.where(norm == np.amin(norm))[0][0]
    cmd_vel = (v_satisfying_constraints[:, min_index])
    if thres == 1:
        cmd_vel = v_desired
    return cmd_vel


def check_constraints(v_sample, Amat, bvec):
    length = np.shape(bvec)[0]

    for i in range(int(length/2)):
        v_sample = check_inside(v_sample, Amat[2*i:2*i+2, :], bvec[2*i:2*i+2])

    return v_sample


def check_inside(v, Amat, bvec):
    v_out = []
    for i in range(np.shape(v)[1]):
        if not ((Amat @ v[:, i] < bvec).all()):
            v_out.append(v[:, i])
    return np.array(v_out).T


def create_constraints(translation, angle, side):
    # create line
    origin = np.array([0, 0, 1])
    point = np.array([np.cos(angle), np.sin(angle)])
    line = np.cross(origin, point)
    line = translate_line(line, translation)

    if side == "left":
        line = line*-1

    A = line[:2]
    b = -line[2]

    return A, b


def translate_line(line, translation):
    matrix = np.eye(3)
    matrix[2, :2] = -translation[:2]
    return matrix @ line

if __name__ == "__main__":
    ind = random.randint(1, 100)
    simulate("velocity_obstacle"+str(ind)+".gif")