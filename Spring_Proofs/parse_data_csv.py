import csv
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
from scipy.signal import medfilt

desired_variables = ['Coriolis', 'Commanded_Torque', 'Jacobian', 'O_T_EE', 'q', 'dq', 'theta', 'dtheta', 'dq_d', 'tau_J']

def parse_franka_state(filename):

    with open(filename, 'r') as file:
        robot_data = dict()
        count = 0
        for line in file:
            
            for var in desired_variables:
                index = line.find("\"" + var + "\"")
                if index > 0:
                    dat_start = line.find('[', index +1)
                    dat_end = line.find(']', dat_start+1)
                    dat = line[dat_start+1:dat_end]

                    if dat[-1] == ',':
                        dat = dat[:-1]

                    if not dat:
                        num_data = []
                    else:
                        try:
                            num_data = [float(x) for x in dat.split(',')]
                        except:
                            num_data = dat

                    if var in robot_data.keys():
                        robot_data[var].append(num_data)
                    else:
                        robot_data[var] = [num_data]
        
        robot_data = add_diff_calculations(robot_data)
        robot_data = add_jac_calculations(robot_data)

    return robot_data

def parse_FT_sensor(filename):

    with open(filename, 'r') as file:
        ft_data = dict()
        ft_data['f_x'] = []
        ft_data['f_y'] = []
        ft_data['f_z'] = []
        ft_data['t_x'] = []
        ft_data['t_y'] = []
        ft_data['t_z'] = []
        count = 0
        first = True
        for line in file:
            if first:
                first = False
                continue
            index = line.find('}",')

            dat = line[index+3:]
            num_data = [float(x) for x in dat.split(',')]

            transformed_forces, transformed_torques = align_ft_frames(num_data)

            ft_data['f_x'].append(transformed_forces[0])
            ft_data['f_y'].append(transformed_forces[1])
            ft_data['f_z'].append(transformed_forces[2])
            ft_data['t_x'].append(transformed_torques[0])
            ft_data['t_y'].append(transformed_torques[1])
            ft_data['t_z'].append(transformed_torques[2])
    
    ft_data = norm_forces(ft_data)
    
    return ft_data

def align_ft_frames(data):

    R_x = np.array([[1,0,0],[0,np.cos(np.radians(180)),-np.sin(np.radians(180))],[0,np.sin(np.radians(180)),np.cos(np.radians(180))]])
    R_z = np.array([[np.cos(np.radians(135)),-np.sin(np.radians(135)),0],[np.sin(np.radians(135)),np.cos(np.radians(135)),0],[0,0,1]])

    R = R_z @ R_x
    R_inv = np.linalg.inv(R)

    forces = np.array(data[:3])
    torques = np.array(data[3:])

    forces_transformed = R_inv @ forces
    torques_transformed = R_inv @ torques

    return forces_transformed, torques_transformed

def norm_forces(data):

    forces_x = np.array(data['f_x'])
    x_avg = np.mean(forces_x[0:100])
    forces_y = np.array(data['f_y'])
    y_avg = np.mean(forces_y[0:100])
    forces_z = np.array(data['f_z'])
    z_avg = np.mean(forces_z[0:100])

    data['f_x_norm'] = forces_x - x_avg
    data['f_y_norm'] = forces_y - y_avg
    data['f_z_norm'] = forces_z - z_avg

    return data

def calculate_velocity_diff(position, delta_t):
    velocities = (position[1:] - position[:-1]) / delta_t
    return velocities

def calculate_acceleration_diff(velocity, delta_t):
    accelerations = (velocity[1:] - velocity[:-1]) / delta_t
    return accelerations

def add_diff_calculations(robot):

    T = robot['O_T_EE']
    positions = np.array([T[i][12:15] for i in range(len(T))])
    velocities = calculate_velocity_diff(positions, 0.001)
    accelerations = calculate_acceleration_diff(velocities, 0.001)

    robot['Cartesian_EE_Velocity_Diff'] = velocities
    robot['Cartesian_EE_Acceleration_Diff'] = accelerations

    return robot

def calculate_joint_acceleration_diff(velocity, delta_t):
    accelerations = []
    for ii in range(velocity.shape[0]-1):
        accel = (velocity[ii] - velocity[ii+1]) / delta_t
        accelerations.append(accel)
    return np.array(accelerations)

def calculate_velocity_jacobian(jacobian, joint_velocities):

    cartesian_velocities = []
    for jacobian_flat, joint_vel in zip(jacobian, joint_velocities):
        jacobian_matrix = np.array(jacobian_flat).reshape(6,7)
        joint_vel_array = np.array(joint_vel)

        cartesian_velocity = jacobian_matrix@joint_vel_array
        cartesian_velocities.append(cartesian_velocity[:3].tolist())
    return cartesian_velocities

def calculate_jacobian_derivative(jacobians, dt):
    jacobian_derivatives = []
    for i in range(1, len(jacobians)):
        jacobian_prev = np.array(jacobians[i - 1]).reshape(6, 7)
        jacobian_curr = np.array(jacobians[i]).reshape(6, 7)
        jacobian_derivative = (jacobian_curr - jacobian_prev) / dt
        jacobian_derivatives.append(jacobian_derivative)
    return jacobian_derivatives

def calculate_acceleration_jacobian(jacobian, joint_velocities):

    jacobian_derivatives = calculate_jacobian_derivative(jacobian, 0.001)

    cartesian_accelerations = []
    for i in range(1, len(jacobian)):
        jac = np.array(jacobian[i]).reshape(6,7)
        jacobian_dot = jacobian_derivatives[i-1]
        joint_vel = np.array(joint_velocities[i])

        # assume joint accelerations are zero
        # accel = J * joint_accel + J_dot * joint_vel
        cartesian_accel = jacobian_dot @ joint_vel
        cartesian_accelerations.append(cartesian_accel)
    
    return np.array(cartesian_accelerations)

def add_jac_calculations(robot):

    robot['Cartesian_EE_Velocity_Jac'] = calculate_velocity_jacobian(robot['Jacobian'], robot['dq'])
    accel = calculate_acceleration_jacobian(robot['Jacobian'], robot['dq'])

    x_filt = medfilt(accel[:,0], 3)
    y_filt = medfilt(accel[:,1], 3)
    z_filt = medfilt(accel[:,2], 3)

    accel_np = np.array([x_filt, y_filt, z_filt]).T
    accel_list = accel_np.tolist()
    robot['Cartesian_EE_Acceleration_Jac'] = accel_list
    return robot


trials = [1,2,3,4]
conditions = ["damping", "friction_compensation", "white_light"]

for tt in trials:
    for con in conditions:

        print(f"Trial {tt}, Condition {con}")

        file_base_r = "./spring_recap/trial_" + str(tt) + "/franka/" + con
        file_base_f = "./spring_recap/trial_" + str(tt) + "/ft/" + con

        robot_file = file_base_r + ".csv"
        franka_info = parse_franka_state(robot_file)
        
        with open(file_base_r + '.pkl', 'wb') as file:
            pickle.dump(franka_info,file)

        ft_file = file_base_f + "_ft.csv"
        ft_info = parse_FT_sensor(ft_file)

        with open(file_base_f + '_ft.pkl', 'wb') as file:
            pickle.dump(ft_info,file)
