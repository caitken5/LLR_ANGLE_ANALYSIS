# This code will be used for producing the difference in angle of average for Force/Velocity Direction Error.
# It will also be used to show the variability in the directed angle. Look for ways to describe the variability of the signal. Frequency analysis?

# TODO: Generate function that calculates average error in this applied force/velocity.
# TODO: Given this angle of error, apply smoothness function done in Google Colab to ensure that there is no jump in
#  the data that would throw off the calculation of variability.
# TODO: Find metric that demonstrates variability of a signal.

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
react_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/8_LLR_ANGLE_GRAPHS/" \
                     "REACTION_TIME"

plot_error_angle = True
plot_react = True
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
                print(file)
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
                    vel = stuff[:, h.data_header.index("Vxy_Mag")]
                    forse = stuff[:, h.data_header.index("Fxy_Mag")]
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
                        plt.suptitle("Error Angles for Force and Velocity " + file_name + "_" + target_num + "_" + to_from_home + "_" + num_prev)
                        ax1 = fig.add_subplot(211)
                        ax2 = fig.add_subplot(212)
                        ax1.grid(visible=True)
                        ax1.set_title("Force")
                        ax1.plot(t, force_theta, label="Force Target Error")
                        ax1.plot(t, f_abs_angle, label="Absolute Force")
                        ax2.set_title("Velocity")
                        ax2.plot(t, velocity_theta, label="Velocity Target Error")
                        ax2.plot(t, v_abs_angle, label="Absolute Velocity")
                        ax1.set_ylabel("Angle [degrees]")
                        ax1.legend()
                        ax2.set_xlabel("Time [s]")
                        ax2.set_ylabel("Angle [degrees]")
                        ax2.legend()
                        save_str = error_angle_folder + '/' + file_name + "_TARGET-" + target_num + "_HOME-" + to_from_home + "_PREV-" + num_prev
                        if save_graphs:
                            plt.savefig(fname=save_str)
                            fig.clf()
                            gc.collect()
                        else:
                            plt.show()
                            plt.close()
                    # Get reaction time index.
                    reaction_time_index = h.reaction_time_index(dist_from_target)
                    v_min_index = h.get_min(vel)
                    f_min_index = h.get_min(forse)
                    # If the minimum of each of these occurs, then get the maximum force value that occurs after this
                    # minimum, and find the first time at which the signal reaches 50% of the maximum value.
                    f_50 = None
                    v_50 = None
                    if v_min_index is not None:
                        v_after = vel[v_min_index:]
                        v_max = np.argmax(v_after)
                        v_50 = h.get_50(v_after, v_max) + v_min_index
                    if f_min_index is not None:
                        f_after = forse[f_min_index:]
                        f_max = np.argmax(f_after)
                        f_50 = h.get_50(f_after, f_max) + f_min_index
                    # Filter the velocity with very low filter.
                    if reaction_time_index is None:
                        print("Could not calculate reaction time for file " + file_name + "for target " + target_num + ", to home is " + to_from_home + ", num prev is " + num_prev)
                    else:
                        # Take the section of data that corresponds to everything after the reaction time.
                        reaction_stuff = stuff[reaction_time_index:, :]
                        # NOTE: Remember when transferring row information from reaction_stuff to stuff that it will be
                        # translated by reaction_time_index.
                        if plot_react:
                            vel = stuff[:, h.data_header.index("Vxy_Mag")]
                            force = stuff[:, h.data_header.index("Fxy_Mag")]
                            # Plot the reaction time and when the first minimum occurs. Can be used to filter out samples
                            # that don't fit the desired trend and might be outlier data.
                            fig = plt.figure(num=1, dpi=100, facecolor='w', edgecolor='w')
                            fig.set_size_inches(25, 8)
                            plt.suptitle("Force and Velocity Reaction Times for " + file_name + "_" + target_num + "_" + to_from_home + "_" + num_prev)
                            ax1 = fig.add_subplot(111)
                            ax1.grid(visible=True)
                            ax1.plot(t, vel*100, label="Velocity [cm/s]")
                            ax1.plot(t, force, label="Force [N]")
                            # Reaction time from Olivia's work does not appear to work as well as liked.
                            if v_min_index is not None:
                                ax1.axvline(x=t[v_min_index], color='g', label="First Minimum Velocity")
                            if v_50 is not None:
                                ax1.axvline(x=t[v_50], color='g', label="Velocity Reaction Time")
                            if f_min_index is not None:
                                ax1.axvline(x=t[f_min_index], color='r', label="First Minimum Force")
                            if f_50 is not None:
                                ax1.axvline(x=t[f_50], color='r', label="Force Reaction Time")
                            ax1.set_xlabel("Time [s]")
                            ax1.set_ylabel("Magnitude")
                            ax1.legend()
                            # Add some stuff to save the file here.
                            save_str = react_folder + '/' + file_name + "_TARGET-" + target_num + "_HOME-" + to_from_home + "_PREV-" + num_prev
                            if save_graphs:
                                plt.savefig(fname=save_str)
                                fig.clf()
                                gc.collect()
                            else:
                                plt.show()
                                plt.close()
                            # Calculate the mean of the force angle and the velocity angle.
                            mean_force_error = h.get_angle_error(force_theta[reaction_time_index:])
                            mean_velocity_error = h.get_angle_error(velocity_theta[reaction_time_index:])

                            # Due to the same problem I was having before with the way the task was designed,
                            # I will have trouble identifying end of first movements since I cannot guarantee
                            # that someone has stopped before they have hit the target. For this reason, I will only
                            # take the total error.
