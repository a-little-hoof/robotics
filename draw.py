'''
Helper funtion that is used to visualize results.
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np

def plot_robot_and_obstacles(robots, obstacles, robot_radius, num_steps, sim_time, filename):
    ROBOT_NUM = len(robots)
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect('equal')
    ax.grid()
    line_list = []
    robot_list = []
    for i in range(ROBOT_NUM):
        robot = robots[i]
        robot_patch = Circle((robot[0, 0], robot[1, 0]),robot_radius, facecolor='green', edgecolor='black')
        robot_list.append(robot_patch)
        line, = ax.plot([], [], '--r')
        line_list.append(line)

    obstacle_list = []
    for obstacle in range(np.shape(obstacles)[2]):
        obstacle = Circle((0, 0), robot_radius,
                          facecolor='aqua', edgecolor='black')
        obstacle_list.append(obstacle)

    def init():
        for i,robot_patch in enumerate(robot_list):
            ax.add_patch(robot_patch)
            line_list[i].set_data([], [])
        for obstacle in obstacle_list:
            ax.add_patch(obstacle)
        return robot_list + line_list + obstacle_list

    def animate(i):
        for num in range(ROBOT_NUM):
            robot = robots[num]
            robot_list[num].center = (robot[0, i], robot[1, i])
            line_list[num].set_data(robot[0, :i], robot[1, :i])
        for j in range(len(obstacle_list)):
            obstacle_list[j].center = (obstacles[0, i, j], obstacles[1, i, j])
        return robot_list + line_list + obstacle_list

    init()
    step = (sim_time / num_steps)
    for i in range(num_steps):
        animate(i)
        plt.pause(step)

    # Save animation

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, num_steps), interval=200,
        blit=True, init_func=init)

    ani.save(filename,fps=15)


