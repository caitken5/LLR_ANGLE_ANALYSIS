import sys
import numpy as np
from scipy import signal

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


def reaction_time_index(dist_from_target):
    # This function calculates the reaction time of the user in this particular reach. It returns the row number that
    # corresponds to when the reaction occurs.
    react_time_index = None
    for j in range(10, dist_from_target.shape[0]):
        row_start = dist_from_target[j] - dist_from_target[j-10]
        if row_start <= -1:
            react_time_index = j
            break
    return react_time_index


def get_reaction_time(data, index):
    # Using the calculated index in get_reaction_time_index, calculate the amount of time from start to when reaction occurs.
    react_time = data[index] - data[0]
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
        theta_arr[i] = angle_between_points(point_1, point_2)*180/np.pi
        # Make sure all values are converted to degrees because it's
        # easier to make sense in the brain what is happening!
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


def butterworth_filter(data, cutoff, fs):
    # 2nd order 50 Hz Butterworth filter applied along the time series dimension of as many columns are passed to it.
    # Will likely just be used to filter the velocity data, since a second order 20Hz Butterworth filter has already
    # been applied to the force data.
    sos = signal.butter(2, cutoff, 'lowpass', fs=fs, output='sos')
    # Frequency was selected to smooth the signal so there are no accidental spikes in data when a person has actually
    # stopped, but not so that it changes the shape of the signal, which if too low makes it difficult to
    # identify stopping regions.
    # Since fs is specified, set the cutoff filter to 20 Hz.
    filtered_data = signal.sosfiltfilt(sos, data, axis=0)
    return filtered_data


def get_min(data):
    # This function takes in the velocity data and calculates whether the first minimum in the data has been achieved.
    # Argrelmin requires a strict minimum, ie. increasing on both sides. Since find_peaks can detect lows that are not
    # strict minimums, I will use that.
    my_min = None  # Base set so I can deal with occasions that don't fit what I expect.
    first_min = []
    # Multiply my series by -1 to find the maxes (which are actually the minima!)
    neg_vel = -1*data
    my_min_series = signal.find_peaks(neg_vel)[0]  # Use 0 to extract the array.
    if my_min_series.size != 0:
        my_min = my_min_series[0]
    return my_min


def get_angle_error(data):
    # Retrieve the mean of the error.
    return np.mean(data)


def get_50(data, my_max):
    my_50 = 0.5 * (data[my_max] - data[0]) + data[0]
    index_50 = None
    # Loop through and find the first sample
    for i in range(data.shape[0]):
        if data[i] >= my_50:
            index_50 = i
            break
    return index_50
