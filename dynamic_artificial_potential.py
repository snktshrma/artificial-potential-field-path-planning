import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

k_att = 4.0
k_rep = 10.0
d0 = 4.0
dt = 0.01
max_rep_force = 14.0
goal = np.array([10, 10])
initial_obstacles = np.array([[3, 3.5], [6, 7], [9, 9], [8, 4], [5, 5]])
obstacle_speeds = np.array([[0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])

def attractive_force(position, goal):
    """
    Calculate the attractive force towards the goal
    """
    return -k_att * (position - goal)

def repulsive_force(position, obstacles, d0, k_rep):
    """
    Calculate the repulsive force from obstacles
    """
    force = np.zeros(2)
    for obs in obstacles:
        diff = position - obs
        dist = np.linalg.norm(diff)
        if dist < d0:
            repulsive_force = k_rep * ((1/dist) - (1/d0)) * (1/(dist**2)) * (diff/dist)
            if np.linalg.norm(repulsive_force) > max_rep_force:
                repulsive_force = repulsive_force / np.linalg.norm(repulsive_force) * max_rep_force
            force += repulsive_force
    return force

def total_force(position, goal, obstacles, d0, k_att, k_rep):
    """
    Calculate the total force on the robot
    """
    force_att = attractive_force(position, goal)
    force_rep = repulsive_force(position, obstacles, d0, k_rep)
    return force_att + force_rep

robot_position = np.array([0, 0])
path_data = [robot_position.copy()]
obstacles = initial_obstacles.copy()

x_range = np.linspace(-5, 15, 50)
y_range = np.linspace(-5, 15, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i, j], Y[i, j]])
        att_potential = 0.5 * k_att * np.linalg.norm(pos - goal)**2
        rep_potential = 0
        for obs in obstacles:
            dist = np.linalg.norm(pos - obs)
            if dist < d0:
                rep_potential += 0.5 * k_rep * ((1/dist) - (1/d0))**2
        Z[i, j] = att_potential + rep_potential
        
        force = total_force(pos, goal, obstacles, d0, k_att, k_rep)
        U[i, j] = force[0]
        V[i, j] = force[1]

fig, ax = plt.subplots()
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
goal_plot, = ax.plot(goal[0], goal[1], 'go', label='Goal')
obstacles_plot, = ax.plot(obstacles[:, 0], obstacles[:, 1], 'ro', label='Obstacles')
path_plot, = ax.plot([], [], 'b-', label='Robot Path')
robot_plot, = ax.plot([], [], 'bo')
potential_contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
quiver_all = ax.quiver(X, Y, U, V, color='white', alpha=0.5)
quiver_robot = ax.quiver([], [], [], [], color='red')

def init():
    path_plot.set_data([], [])
    robot_plot.set_data([], [])
    quiver_robot.set_UVC([], [])
    return path_plot, robot_plot, quiver_robot

def update(frame):
    global robot_position, obstacles
    force = total_force(robot_position, goal, obstacles, d0, k_att, k_rep)
    robot_position = robot_position + force * dt
    path_data.append(robot_position.copy())

    obstacles[:, 0] += obstacle_speeds[:, 0] * np.sin(0.05 * frame)
    obstacles[:, 1] += obstacle_speeds[:, 1] * np.cos(0.05 * frame)
    
    for obs_idx, obs in enumerate(obstacles):
        if obs[0] < -5 or obs[0] > 15:
            obstacles[obs_idx, 0] = np.clip(obs[0], -5, 15)
            obstacle_speeds[obs_idx, 0] *= -1
        if obs[1] < -5 or obs[1] > 15:
            obstacles[obs_idx, 1] = np.clip(obs[1], -5, 15)
            obstacle_speeds[obs_idx, 1] *= -1

    path = np.array(path_data)
    path_plot.set_data(path[:, 0], path[:, 1])
    robot_plot.set_data(robot_position[0], robot_position[1])
    obstacles_plot.set_data(obstacles[:, 0], obstacles[:, 1])

    quiver_robot.set_offsets(robot_position)
    quiver_robot.set_UVC(force[0], force[1])

    return path_plot, robot_plot, obstacles_plot, quiver_robot

ani = animation.FuncAnimation(fig, update, frames=2000, init_func=init, blit=True, interval=50, repeat=False)

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Dynamic Potential Field')
ax.grid()
plt.show()