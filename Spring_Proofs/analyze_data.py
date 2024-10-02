import matplotlib.pyplot as plt
import numpy as np
import pickle


def magnitudes(x,y,z):

    mags = []
    for ii in range(x.shape[0]):
        mag = np.sqrt(x[ii]*x[ii] + y[ii]*y[ii] + z[ii]*z[ii])
        mags.append(mag)
    return np.array(mags)

def find_spike(data, threshold=0.015):
    s1 = np.where(abs(data[:,0]) > threshold)[0][0]
    s2 = np.where(abs(data[:,1]) > threshold)[0][0]
    s3 = np.where(abs(data[:,2]) > threshold)[0][0]

    return np.min([s1,s2,s3])

def alignment_indices(velocities, acceleration, forces):
    franka_spike = find_spike(velocities,0.015)
    force_spike = find_spike(forces,0.50)

    shift = force_spike - franka_spike

    if shift > 0:
        min_length = min(forces.shape[0]-shift, acceleration.shape[0])
        force_ind = [shift, min_length+shift]
        franka_ind = [0,min_length]
    else:
        min_length = min(acceleration.shape[0]+shift, forces.shape[0])
        franka_ind = [-shift, min_length-shift]
        force_ind = [0, min_length]
    
    return force_ind, franka_ind

def calulate_mc_coeef(velocity, acceleration, force):

    mat_av = np.array([acceleration, -1*velocity]).T
    force = force.reshape(-1,1)
    inv_mat = np.linalg.pinv(mat_av)
    test_mc = inv_mat @ force
    
    return test_mc


##### Load all Conditions ######
all_data = dict()

trials = [1,2,3,4]
for ii in trials:
    # Load Damping data
    file_base = './spring_recap/trial_' + str(ii)
    with open(file_base + '/ft/damping_ft.pkl', 'rb') as file:
        f1 = pickle.load(file)

    with open(file_base + '/franka/damping.pkl', 'rb') as file:
        frank1 = pickle.load(file)

    # Load Friction Comp Data
    with open(file_base + '/ft/friction_compensation_ft.pkl', 'rb') as file:
        f2 = pickle.load(file)

    with open(file_base + '/franka/friction_compensation.pkl', 'rb') as file:
        frank2 = pickle.load(file)

    # Load White Light Data
    with open(file_base + '/ft/white_light_ft.pkl', 'rb') as file:
        f3 = pickle.load(file)

    with open(file_base + '/franka/white_light.pkl', 'rb') as file:
        frank3 = pickle.load(file)

    all_data[ii] = dict(Damping=dict(franka=frank1, ft=f1), Friction_Compensation=dict(franka=frank2, ft=f2), White_Light=dict(franka=frank3, ft=f3))


show_3d = False
show_2d = False
trials = list(all_data.keys())
conditions = list(all_data[1].keys())
# mode = 'Damping'

##### Loop over all conditions #######
if show_2d:
    # initialize plots
    # Velocity vs Force Plots
    fig1, ax1 = plt.subplots(1,4)
    fig1.suptitle('X Axis')
    fig2, ax2 = plt.subplots(1,4)
    fig2.suptitle('Y Axis')
    fig3, ax3 = plt.subplots(1,4)
    fig3.suptitle('Z Axis')
    fig4, ax4 = plt.subplots(1,4)
    fig4.suptitle('Magnitude')

    # Force and Velocity Plots
    fig6, ax6 = plt.subplots(1,4)
    fig6.suptitle('X Axis')
    fig7, ax7 = plt.subplots(1,4)
    fig7.suptitle('Y Axis')
    fig8, ax8 = plt.subplots(1,4)
    fig8.suptitle('Z Axis')
    fig9, ax9 = plt.subplots(1,4)
    fig9.suptitle('Magnitude')

for mode in conditions:
    for ii in range(len(trials)):

        print(f'{mode} Condition, Trial {trials[ii]}')

        franka_data = all_data[trials[ii]][mode]['franka']
        ft_data = all_data[trials[ii]][mode]['ft']

        vel = np.array(franka_data['Cartesian_EE_Velocity_Jac'])
        accel = np.array(franka_data['Cartesian_EE_Acceleration_Jac'])
        forces = np.array([ft_data['f_x_norm'], ft_data['f_y_norm'], ft_data['f_z_norm']]).T

        force_idx, franka_idx = alignment_indices(vel, accel, forces)

        aligned_forces = forces[force_idx[0]:force_idx[1], :]
        aligned_vel = vel[franka_idx[0]:franka_idx[1], :]
        aligned_accel = accel[franka_idx[0]:franka_idx[1], :]

        # f = m*a - c*v
        # f = [a -v][[m] [c]]
        # inv([a -v])*[f] = [[m] [c]]

        # x
        # mat_av_x = np.array([aligned_accel[:,0], -1*aligned_vel[:,0]]).T
        # force_x = aligned_forces[:,0].reshape(-1,1)
        # inv_mat = np.linalg.pinv(mat_av_x)
        # test_mc = inv_mat @ force_x
        # print(test_mc)

        mc_x = calulate_mc_coeef(aligned_vel[:,0], aligned_accel[:,0], aligned_forces[:,0])
        mc_y = calulate_mc_coeef(aligned_vel[:,1], aligned_accel[:,1], aligned_forces[:,1])
        mc_z = calulate_mc_coeef(aligned_vel[:,2], aligned_accel[:,2], aligned_forces[:,2])

        print(f'X: m = {mc_x[0]}, c = {mc_x[1]}')
        print(f'Y: m = {mc_y[0]}, c = {mc_y[1]}')
        print(f'Z: m = {mc_z[0]}, c = {mc_z[1]}')

        # force_mag = magnitudes(aligned_forces[:,0], aligned_forces[:,1], aligned_forces[:,2])
        # vel_mag = magnitudes(aligned_vel[:,0], aligned_vel[:,1], aligned_vel[:,2])
        # accel_mag = magnitudes(aligned_accel[:,0], aligned_accel[:,1], aligned_accel[:,2])


    if show_2d:
        # X
        ax1[ii].scatter(aligned_vel[:,0], aligned_forces[:,0], s=5, alpha=0.5)
        ax1[ii].set_title(trials[ii])
        ax1[ii].set_xlim([-0.4,0.4])
        ax1[ii].set_ylim([-20.0,20.0])
        ax1[ii].set_ylabel('Force (N)')
        # Y
        ax2[ii].scatter(aligned_vel[:,1], aligned_forces[:,1], s=5, alpha=0.5)
        ax2[ii].set_title(trials[ii])
        ax2[ii].set_xlim([-0.4,0.4])
        ax2[ii].set_ylim([-20.0,20.0])
        ax2[ii].set_ylabel('Force (N)')
        # Z
        ax3[ii].scatter(aligned_vel[:,2], aligned_forces[:,2], s=5, alpha=0.5)
        ax3[ii].set_title(trials[ii])
        ax3[ii].set_xlim([-0.4,0.4])
        ax3[ii].set_ylim([-20.0,20.0])
        ax3[ii].set_ylabel('Force (N)')
        # Magnitude
        ax4[ii].scatter(vel_mag, force_mag, s=5, alpha=0.5)
        ax4[ii].set_title(trials[ii])
        ax4[ii].set_xlim([-0.025,0.4])
        ax4[ii].set_ylim([-1.0,25.0])
        ax4[ii].set_ylabel('Force (N)')

        
        line1, = ax6[ii].plot(aligned_vel[:,0], color='b')
        ax6[ii].set_ylabel('Velocity', color='k')
        # ax6[0].set_xlabel('Time (ms)')
        ax6[ii].set_xticklabels([])
        ax10 = ax6[ii].twinx()
        line2, = ax10.plot(aligned_forces[:,0], color='r')
        ax10.set_ylabel('Force', color='k')
        ax6[ii].set_ylim([-0.5,0.5])
        ax10.set_ylim([-20.0,20.0])
        ax6[ii].legend([line1, line2], ['Velocity', 'Force'], loc='upper right')
        ax6[ii].set_title(trials[ii])

        line1, = ax7[ii].plot(aligned_vel[:,1], color='b')
        ax7[ii].set_ylabel('Velocity', color='k')
        # ax6[1].set_xlabel('Time (ms)')
        ax7[ii].set_xticklabels([])
        ax11 = ax7[ii].twinx()
        line2, = ax11.plot(aligned_forces[:,1], color='r')
        ax11.set_ylabel('Force', color='k')
        ax7[1].set_ylim([-0.5,0.5])
        ax11.set_ylim([-20.0,20.0])
        ax7[1].legend([line1, line2], ['Velocity', 'Force'], loc='upper right')
        ax7[1].set_title(trials[ii])

        line1, = ax8[ii].plot(aligned_vel[:,2], color='b')
        ax8[ii].set_ylabel('Velocity', color='k')
        # ax6[2].set_xlabel('Time (ms)')
        ax8[ii].set_xticklabels([])
        ax12 = ax8[ii].twinx()
        line2, = ax12.plot(aligned_forces[:,2], color='r')
        ax12.set_ylabel('Force', color='k')
        ax8[ii].set_ylim([-0.5,0.5])
        ax12.set_ylim([-20.0,20.0])
        ax8[ii].legend([line1, line2], ['Velocity', 'Force'], loc='upper right')
        ax8[ii].set_title(trials[ii])

        line1, = ax9[ii].plot(vel_mag, color='b')
        ax9[ii].set_ylabel('Velocity', color='k')
        # ax9[ii].set_xlabel('Time (ms)')
        ax13 = ax9[ii].twinx()
        line2, = ax13.plot(force_mag, color='r')
        ax13.set_ylabel('Force', color='k')
        ax9[ii].set_ylim([0.0,0.5])
        ax13.set_ylim([0.0,25.0])
        ax9[ii].legend([line1, line2], ['Velocity', 'Force'], loc='upper right')
        ax9[ii].set_title(trials[ii])

        if ii == 2:
            ax1[ii].set_xlabel('Velocity')
            ax2[ii].set_xlabel('Velocity')
            ax3[ii].set_xlabel('Velocity')
            ax4[ii].set_xlabel('Velocity')
            ax6[ii].set_xlabel('Time (ms)')
            ax7[ii].set_xlabel('Time (ms)')
            ax8[ii].set_xlabel('Time (ms)')
            ax9[ii].set_xlabel('Time (ms)')


    if show_3d:

        fig_name = 'Trial ' + str(trials[ii]) + ' X axis'
        fig14 = plt.figure(fig_name)
        ax14 = fig14.add_subplot(111, projection='3d')
        scatter = ax14.scatter(aligned_vel[::10,0], aligned_accel[::10,0], aligned_forces[::10,0], alpha=0.6)
        ax14.set_xlabel('Velocity')
        ax14.set_ylabel('Acceleration')
        ax14.set_zlabel('Force')
        ax14.set_xlim([-0.4,0.4])
        ax14.set_ylim([-0.4,0.4])
        ax14.set_zlim([-25.0,25.0])
        ax14.set_title('X axis')

        fig_name = 'Trial ' + str(trials[ii]) + ' Y axis'
        fig15 = plt.figure(fig_name)
        ax15 = fig15.add_subplot(111, projection='3d')
        scatter = ax15.scatter(aligned_vel[::10,1], aligned_accel[::10,1], aligned_forces[::10,1], alpha=0.6)
        ax15.set_xlabel('Velocity')
        ax15.set_ylabel('Acceleration')
        ax15.set_zlabel('Force')
        ax15.set_xlim([-0.4,0.4])
        ax15.set_ylim([-0.4,0.4])
        ax15.set_zlim([-25.0,25.0])
        ax15.set_title('Y axis')

        fig_name = 'Trial ' + str(trials[ii]) + ' Z axis'
        fig16 = plt.figure(fig_name)
        ax16 = fig16.add_subplot(111, projection='3d')
        scatter = ax16.scatter(aligned_vel[::10,2], aligned_accel[::10,2], aligned_forces[::10,2], alpha=0.6)
        ax16.set_xlabel('Velocity')
        ax16.set_ylabel('Acceleration')
        ax16.set_zlabel('Force')
        ax16.set_xlim([-0.4,0.4])
        ax16.set_ylim([-0.4,0.4])
        ax16.set_zlim([-25.0,25.0])
        ax16.set_title('Z axis')

        fig_name = 'Trial ' + str(trials[ii]) + ' Magnitude'
        fig17 = plt.figure(fig_name)
        ax17 = fig17.add_subplot(111, projection='3d')
        scatter = ax17.scatter(vel_mag[::10], accel_mag[::10], force_mag[::10], alpha=0.6)
        ax17.set_xlabel('Velocity')
        ax17.set_ylabel('Acceleration')
        ax17.set_zlabel('Force')
        ax17.set_xlim([0.0,0.4])
        ax17.set_ylim([0.0,0.4])
        ax17.set_zlim([0.0,25.0])
        ax17.set_title('Magnitude')

        fig_name = 'Trial ' + str(trials[ii]) + ' Velocity vs Force'
        fig18 = plt.figure(fig_name)
        ax18 = fig18.add_subplot(111, projection='3d')
        scatter = ax18.scatter(aligned_vel[::10,0], aligned_vel[::10,1], aligned_vel[::10,2], c=force_mag[::10], alpha=0.6, cmap='viridis')
        color_bar = plt.colorbar(scatter, ax=ax18, shrink=0.5, aspect=5)
        color_bar.set_label('Force')
        ax18.set_xlabel('X Velocity')
        ax18.set_ylabel('Y Velocity')
        ax18.set_zlabel('Z Velocity')
        # ax18.set_xlim([0.0,0.4])
        # ax18.set_ylim([0.0,0.4])
        # ax18.set_zlim([0.0,25.0])
        ax18.set_title('Velocity Direction vs Force Magnitude')


# Show the plot
# plt.show()








