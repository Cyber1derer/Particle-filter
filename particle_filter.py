import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data

# add random seed for generating comparable pseudo random numbers
np.random.seed(123)

# plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.

    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx = []
    ly = []

    for i in range(len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy', scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)


def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles


def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        # make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    # calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

import math as m

def _wrap_to_pi(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def sample_motion_model(odometry, particles):
    """Sample new particle positions using the odometry motion model."""

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # motion noise parameters [alpha1, alpha2, alpha3, alpha4]
    alpha1, alpha2, alpha3, alpha4 = [0.1, 0.1, 0.05, 0.05]

    new_particles = []


    '''your code here'''
    assert delta_trans >=0
    ##from Lab4
    def sample_motion_model2(X, U, Alpha):
        sample = lambda sigma: np.random.normal(0,sigma)
        x,y,theta = X

        delta_r1, delta_r2, delta_t = U
        dr1,dr2,d_t = delta_r1,delta_r2,delta_t

        alpha_1, alpha_2, alpha_3, alpha_4 = Alpha
        a1,a2,a3,a4 = alpha_1,alpha_2,alpha_3,alpha_4

        # apply noise following the odometry motion model from
        # "Probabilistic Robotics" (Eq. 5.13)
        delta_dot_r1 = dr1 + sample(a1 * abs(dr1) + a2 * d_t)
        delta_dot_r2 = dr2 + sample(a1 * abs(dr2) + a2 * d_t)
        delta_dot_t = d_t + sample(a3 * d_t + a4 * (abs(dr1) + abs(dr2)))

        x_dot = x + delta_dot_t*m.cos(theta+delta_dot_r1)
        y_dot = y + delta_dot_t*m.sin(theta+delta_dot_r1)
        theta_dot = theta + delta_dot_r1+delta_dot_r2
        return x_dot, y_dot, theta_dot
    
    # particle_m1 = dict()
    # for particle in particles:
        
    #     particle_m = [particle['x'],particle['y'],particle['theta']]
    #     for i in range(delta_rot1.shape[0]):
    #         od = [delta_rot1[i],delta_trans[i],delta_rot2[i]]
    #         particle_m = sample_motion_model2(particle_m,od, noise)
        
        
    #     particle_m1['x'],particle_m1['y'],particle_m1['theta'] = particle_m[0],particle_m[1],particle_m[2]
    #     new_particles.append(particle_m1)


    #оказывается odometry это одно измерение,а не серия
    particle_m = dict()
    for particle in particles:
        particle_m = [particle['x'], particle['y'], particle['theta']]
        #od = [delta_rot1,delta_trans,delta_rot2]
        od = [delta_rot1,delta_rot2,delta_trans]
        particle_new = sample_motion_model2(particle_m,od, noise)
        particle_new_dict = {'x': particle_new[0], 'y': particle_new[1], 'theta': particle_new[2]}
        new_particles.append(particle_new_dict)


        x = particle['x'] + trans_hat * np.cos(particle['theta'] + rot1_hat)
        y = particle['y'] + trans_hat * np.sin(particle['theta'] + rot1_hat)
        theta = _wrap_to_pi(particle['theta'] + rot1_hat + rot2_hat)

        new_particles.append({'x': x, 'y': y, 'theta': theta})

    return new_particles


def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the particle and landmark positions and sensor measurements
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []

    for particle in particles:
        weight = 1.0
        for lm_id, r_meas in zip(ids, ranges):
            landmark_pos = landmarks[lm_id]
            dist = np.linalg.norm([particle['x'] - landmark_pos[0],
                                   particle['y'] - landmark_pos[1]])
            weight *= scipy.stats.norm(loc=dist, scale=sigma_r).pdf(r_meas)
        weights.append(weight)

    # normalize weights
    normalizer = sum(weights)
    weights = (np.array(weights) / normalizer).tolist()

    return weights


def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    M = len(particles)
    step = 1.0 / M
    r = np.random.uniform(0, step)
    c = weights[0]
    i = 0
    for m in range(M):
        U = r + m * step
        while U > c:
            i += 1
            c += weights[i]
        new_particles.append({
            'x': particles[i]['x'],
            'y': particles[i]['y'],
            'theta': particles[i]['theta']
        })

    return new_particles


def main():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("data/sensor_data.dat")

    # initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    # run particle filter
    for timestep in range(len(sensor_readings) // 2):

        # plot the current state
        plot_state(particles, landmarks, map_limits)

        # predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)

        # resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

    plt.show(block=True)


if __name__ == "__main__":
    main()
