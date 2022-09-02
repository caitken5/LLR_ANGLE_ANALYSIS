# This code will be used for producing the difference in angle of average for Force/Velocity Direction Error.
# It will also be used to show the variability in the directed angle. Look for ways to describe the variability of the signal. Frequency analysis?

# TODO: Determine if this should be done for the entire sequence, or if this should be done for only after the reaction
#  time (that code can be fed in to demonstrate what is happening). I think after reaction time makes sense.
# TODO: Generate function that calculates average error in this applied force/velocity.

# TODO: Given this angle of error, apply smoothness function done in Google Colab to ensure that there is no jump in
#  the data that would throw off the calculation of variability.
# TODO: Find metric that demonstrates variability of a signal. Due to small samples,
#  performing frequency analysis on this would be difficult.

# TODO: May want to insert the code for the calculation of the angle difference for both force and velocity into the
#  data cleaning code, since this can be treated as an additional signal.

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gc

import header as h

source_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/" \
                "4_LLR_DATA_SEGMENTATION/NPZ_FILES_BY_TARGET"
error_angle_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/8_LLR_ANGLE_GRAPHS/" \
                     "ERROR_ANGLE"
reaction_and_end_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/" \
                          "8_LLR_ANGLE_GRAPHS/REACTION_AND_END"

plot_error_angle = True
plot_react_and_min = True
save_graphs = True

if __name__ == '__main__':
    print("Running angle analysis...")
    for file in os.listdir(source_folder):
        if save_graphs:
            matplotlib.use('Agg')
        if file.endswith('.npz'):
            task_number = h.get_task_number(file)
            if (("T1" in file) or ("T2" in file)) & ("V0" in file):
                # Here if file includes task 1 and 2, and if it also is a V0 file, as I'm not studying other pieces
                # of information.
                # print(file)
                source_file = source_folder + '/' + file
                file_name = file.split('.')[0]
                # The source file is a npz file. So I need to load in the data and unpack.
                data = np.load(source_file, allow_pickle=True)
                # Unpack the data.
                ragged_list, target_list, target_i = h.load_npz(data)
                # Loop through the variables stored in ragged_list.
                for i in range(len(ragged_list)):
                    # print(i)
                    stuff = ragged_list[i]
                    # Extract important pieces of information about target number, to from home, and trial number.
                    target_num = str(int(stuff[0, h.data_header.index("Target_Num")]))
                    to_from_home = str(int(stuff[0, h.data_header.index("To_From_Home")]))
                    num_prev = str(int(stuff[0, h.data_header.index("Num_Prev_Targets")]))
                    # Extract the relevant columns of information for force, velocity, and target location.
                    t = stuff[:, h.data_header.index("Time")]
                    force = stuff[:, [h.data_header.index("CorrForce_X"), h.data_header.index("CorrForce_Y")]]
                    velocity = stuff[:, [h.data_header.index("X_Vel"), h.data_header.index("Y_Vel")]]
                    des_pos = stuff[:, [h.data_header.index("Des_X_Pos"), h.data_header.index("Des_Y_Pos")]]
                    f_abs_angle = stuff[:, h.data_header.index("Fxy_Angle")]
                    v_abs_angle = stuff[:, h.data_header.index("Vxy_Angle")]
                    dist_from_target = stuff[:, h.data_header.index("Dist_From_Target")]
                    if (des_pos[0, 0] == 0) and (des_pos[0, 1] == 0):
                        # Based on the task number, calculate the new vector.
                        des_pos = np.ones((stuff.shape[0], 2))
                        if (i == 0) and (stuff[0, h.data_header.index("Target_Num")] == 1):
                            # Here if in the first iteration of the data and the target_num is 1.
                            des_pos *= [-31.5, 0]
                        else:
                            des_pos *= ragged_list[i-1][0, [h.data_header.index("Des_X_Pos"), h.data_header.index("Des_Y_Pos")]].tolist()
                    force_theta = h.angle_between_vectors(force, des_pos)
                    velocity_theta = h.angle_between_vectors(velocity, des_pos)
                    # NOW plot briefly to also observe if there are inconsistencies in the recorded signal that need
                    # to be altered for smoother plotting.
                    if plot_error_angle:
                        # Plot the thetas!
                        fig = plt.figure(num=1, dpi=100, facecolor='w', edgecolor='w')
                        fig.set_size_inches(25, 8)
                        ax1 = fig.add_subplot(211)
                        ax2 = fig.add_subplot(212)
                        ax1.grid(visible=True)
                        ax1.plot(t, force_theta, label="Force Target Error")
                        ax1.plot(t, f_abs_angle, label="Absolute Force")
                        ax2.plot(t, velocity_theta, label="Velocity Target Error")
                        ax2.plot(t, v_abs_angle, label="Absolute Velocity")
                        ax1.set_ylabel("Angle [rad]")
                        ax1.legend()
                        ax2.set_xlabel("Time [s]")
                        ax2.set_ylabel("Angle [rad]")
                        ax2.legend()
                        save_str = error_angle_folder + '/' + file_name + "_TARGET-" + target_num + "_HOME-" + to_from_home + "_PREV-" + num_prev
                        if save_graphs:
                            plt.savefig(fname=save_str)
                            fig.clf()
                            gc.collect()
                        else:
                            plt.show()
                    # Get reaction time index.
                    reaction_time_index = h.reaction_time_index(dist_from_target)
                    # Filter the velocity with very low filter.
                    if reaction_time_index is None:
                        print("Could not calculate reaction time for file " + file_name + "for target " +  target_num + ", to home is " + to_from_home + ", num prev is " + num_prev)
                    else:
                        # Take the section of data that corresponds to everything after the reaction time.
                        reaction_stuff = stuff[reaction_time_index:, :]
                        # NOTE: Remember when transferring row information from reaction_stuff to stuff that it will be
                        # translated by reaction_time_index.
                        vel = stuff[:, h.data_header.index("Vxy_Mag")]
                        Vxy_Mag = reaction_stuff[:, h.data_header.index("Vxy_Mag")]
                        vf = h.butterworth_filter(Vxy_Mag, 3, 100)
                        vel_min_1 = h.get_vel_min(vf)
                        if vel_min_1 is None:
                            print("Could not calculate end of first movement for file " + file_name + "for target "
                                  + target_num + ", to home is " + to_from_home + ", num prev is " + num_prev)
                        else:
                            vel_min_1 = h.get_vel_min(vf) + reaction_time_index
                            if plot_react_and_min:
                                # Plot the reaction time and when the first minimum occurs. Can be used to filter out samples
                                # that don't fit the desired trend and might be outlier data.
                                fig = plt.figure(num=1, dpi=100, facecolor='w', edgecolor='w')
                                fig.set_size_inches(25, 8)
                                plt.suptitle("Velocity, Reaction Time, and End of First Movement for " + file_name)
                                ax1 = fig.add_subplot(111)
                                ax1.grid(visible=True)
                                ax1.plot(t, vel, label="Velocity [m/s]")
                                ax1.axvline(x=t[reaction_time_index], color='g', label="Reaction Time")
                                ax1.axvline(x=t[vel_min_1], color='r', label="First Minimum")
                                ax1.set_xlabel("Time [s]")
                                ax1.legend()
                                # Add some stuff to save the file here.
                                save_str = reaction_and_end_folder + '/' + file_name + "_TARGET-" + target_num + "_HOME-" + to_from_home + "_PREV-" + num_prev
                                if save_graphs:
                                    plt.savefig(fname=save_str)
                                    fig.clf()
                                    gc.collect()
                                else:
                                    plt.show()
                                    plt.close()
                            # TODO: Calculate error of angle for force and velocity from reaction time until end.

                            # Due to the same problem I was having before with the way the task was designed,
                            # I will have trouble identifying end of first movements since I cannot guarantee
                            # that someone has stopped before they have hit the target. For this reason, I will only
                            # take the total error.
                            # TODO: Calculate error of angle for force and velocity from reaction time to end of first movement.
