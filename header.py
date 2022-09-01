import sys
import numpy as np

data_header = ['Time', 'Des_X_Pos', 'Des_Y_Pos', 'X_Pos', 'Y_Pos', 'OptoForce_X', 'OptoForce_Y', 'OptoForce_Z',
               'OptoForce_Z_Torque', 'Theta_1', 'Theta_2', 'Fxy_Mag', 'Fxy_Angle', 'CorrForce_X', 'CorrForce_Y',
               'Target_Num', 'X_Vel', 'Y_Vel', 'Vxy_Mag', 'Vxy_Angle', 'Dist_From_Target', 'Disp_Btw_Pts', 'Est_Vel',
               'To_From_Home', 'Num_Prev_Targets', 'Resistance', 'dFxy_Mag', 'dVxy_Mag', 'Fxy_Angle_Smooth',
               'dFxy_Angle_Smooth', 'Vxy_Angle_Smooth', 'dVxy_A_Smooth']


def load_npz(npz_file):
    # This function loads npz file, and reconstructs a ragged list of numpy arrays given data and counter.
    # Returns ragged_list, data, and counter.
    # Assume that each of the npz files contains two pieces of information: the data to be unpacked and the unpacking indices.
    my_list = npz_file.files
    data = npz_file[my_list[0]]
    counter = npz_file[my_list[1]]
    sep_row = np.cumsum(counter)
    # Define an object that the arrays can be loaded into, noting that they are ragged because the number of rows may differ each time.
    ragged_list = [data[0:sep_row[0], :]]  # Append the first set of data.
    for j in range(len(counter)-1):
        ragged_list.append(data[sep_row[j]:sep_row[j+1], :])
        # Since the list values for each array is calculated, the last row in the array is included in the calculation
        # and a separate line to append the last piece of data from a start to end point is not needed.
    return ragged_list, data, counter


def get_task_number(file_name):
    if "T1" in file_name:
        task = 1
    elif "T2" in file_name:
        task = 2
    elif "T3" in file_name:
        task = 3
    elif "T4" in file_name:
        task = 4
    else:
        my_str = "Task name wasn't found, file_name: " + file_name + ", exiting program."
        sys.exit(my_str)
    return task


def reaction_time(dist_from_target):
    # This function calculates the reaction time of the user in this particular reach. It returns the row number that
    # corresponds to when the reaction occurs.
    react_time = None
    for j in range(10, dist_from_target.shape[0]):
        row_start = dist_from_target[j] - dist_from_target[j-10]
        if row_start <= -1:
            react_time = j
            break
    return react_time


def angle_between_vectors(vector_1, vector_2):
    # Both vector_1 and vector_2 need to be the same length. For each time sample, calculate the angle between the
    # vectors passed.
    len_1 = vector_1.shape[0]
    len_2 = vector_2.shape[0]
    if len_1 != len_2:
        print("These two variables are not the same shape, error incoming!")
    theta_arr = np.zeros((len_1, 1))
    for i in range(len_1):
        point_1 = vector_1[i, :]
        point_2 = vector_2[i, :]
        theta_arr[i] = angle_between_points(point_1, point_2)
    return theta_arr


def angle_between_points(point_1, point_2):
    # Calculates the angle between two points.
    # EQUATION - theta = cos-1((point_1*point_2)/(|point_1|*|point_2|)
    dot = point_1 @ point_2
    mag_1 = np.linalg.norm(point_1)
    mag_2 = np.linalg.norm(point_2)
    var = dot/(mag_1*mag_2)
    # TODO: Also consider positive and negative angle.
    theta = np.arccos(var)
    return theta


