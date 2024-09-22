# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:26:23 2023

@author: akmal
"""

import numpy as np
import matplotlib.pyplot as plt
import glob


def main():
    filepath = r"C:/Users/akmal/OneDrive/Desktop/DATA FOR A WEEK/18 sept/stare6"
    data_files, num_files = read_file(filepath)
    print(num_files)
    
    data_list1, data_list2 = read_data(data_files)
    
    # Checking the shape of the list
    # the list shape: (282, 5) list 1, (282, 1800) for list 2
    
    # Understanding the shape of the list
    # rows = len(data_list2)
    # cols = max(len(sublist) for sublist in data_list2)  # Get the maximum sublist length
    # shape = (rows, cols)
    # print(shape)
    
    # # Compute and print statistics for Doppler data 
    # data_1 = data_list2[0]
    # doppler = [float(row[1]) for row in data_1]
    # dop_mean, dop_var, dop_stdev = get_stats(doppler)
    # print("Doppler Mean:", dop_mean)
    # print("Doppler Variance:", dop_var)
    # print(f"Doppler Standard Deviation: {dop_stdev}\n")
    
    # Taking the altitude and time serries of the backscatter data
    metadata = read_metadata(data_files) # data shape (7, 21)
    gate_length = metadata[2]
    gate_length_float = [float(x) for x in gate_length]
    
    
    # Extracting range gate from data list 2
    all_range_gate = extracting_data(0, data_list2)
        
    # Extracting backscatter
    beta = extracting_data(3, data_list2)
    all_backscatter = [item for sublist in beta for item in sublist]
    
    # Extracting intensity (SNR + 1)
    intensity_list = extracting_data(2, data_list2)
    all_intensity = [item for sublist in intensity_list for item in sublist]
    
    # Extracting SNR
    SNR_list = [[item - 1 for item in sublist] for sublist in intensity_list]
    all_SNR = [item for sublist in SNR_list for item in sublist]
    
    # Extracting doppler data
    doppler_list = extracting_data(1, data_list2)
    all_doppler = [item for sublist in doppler_list for item in sublist]
    
    # Finding the altitude
    list_altitude = get_altitude(all_range_gate, gate_length_float[0])
    all_altitude = [item for sublist in list_altitude for item in sublist]
    
    # Extracting the decimal time
    list_decimal_time = []
    for row in data_list1:
        list_decimal_time.append(row[0])
        
    duplicated_decimal_time = [item for item in list_decimal_time for _ in range(1800)]
    float_dec_time = [float(item) for item in duplicated_decimal_time]
    
    ## Filtering the data using the upper and lower boundary
    k = 1 # Stdev multiplier
    
    # Backscatter filter
    beta_mean, beta_var, beta_stdev = get_stats(all_backscatter)
    lower_bound_beta = beta_mean - k * beta_stdev
    upper_bound_beta = beta_mean + k * beta_stdev
    print(f"For backscatter,the upper bound is {upper_bound_beta:.3f} and the lower bound is {lower_bound_beta:.3f} within {k} standard deviation range.\n")
    
    filtered_backscatter, beta_index = data_filter(all_backscatter, upper_bound_beta, lower_bound_beta) 
    beta_altitude = metadata_filter(all_altitude, beta_index)
    beta_time = metadata_filter(float_dec_time, beta_index)
    
    # Intensity Filter
    int_mean, int_var, int_stdev = get_stats(all_intensity)
    lower_bound_int = int_mean - k * int_stdev
    upper_bound_int = int_mean + k * int_stdev
    print(f"For intensity (SNR+1),the upper bound is {upper_bound_int:.3f} and the lower bound is {lower_bound_int:.3f} within {k} standard deviation range.\n")
    
    filtered_intensity, filtered_index = data_filter(all_intensity, upper_bound_int, lower_bound_int) 
    int_altitude = metadata_filter(all_altitude, filtered_index)
    int_time = metadata_filter(float_dec_time, filtered_index)
    
    # SNR Filter
    SNR_mean, SNR_var, SNR_stdev = get_stats(all_SNR)
    lower_bound_SNR = SNR_mean - k * SNR_stdev
    upper_bound_SNR = SNR_mean + k * SNR_stdev
    print(f"For SNR,the upper bound is {upper_bound_SNR:.3f} and the lower bound is {lower_bound_SNR:.3f} within {k} standard deviation range.\n")
    
    filtered_SNR, SNR_index = data_filter(all_SNR, upper_bound_SNR, lower_bound_SNR) 
    SNR_altitude = metadata_filter(all_altitude, SNR_index)
    SNR_time = metadata_filter(float_dec_time, SNR_index)
    
    # Doppler velocity filter
    dop_mean, dop_var, dop_stdev = get_stats(all_doppler)
    lower_bound_dop = dop_mean - k * dop_stdev
    upper_bound_dop = dop_mean + k * dop_stdev
    print(f"For doppler, the upper bound is {upper_bound_dop:.3f} and the lower bound is {lower_bound_dop:.3f} within {k} standard deviation range.\n")
    
    filtered_doppler, dop_index = data_filter(all_doppler, upper_bound_dop, lower_bound_dop) 
    dop_altitude = metadata_filter(all_altitude, dop_index)
    dop_time = metadata_filter(float_dec_time, dop_index)
    
    
    
    # 1st plot altitude vs backscatter for all data
    graph_plot(all_backscatter, all_altitude, 'attenuated backscatter', 'Altitude (m)', 'Backscatter vs altitude (for all data)')
    
    # 2nd plot altitude vs backscatter for all filtered data
    graph_plot(filtered_backscatter, beta_altitude, 'attenuated backscatter', 'Altitude (m)', 'Backscatter vs altitude (for filtered data)')
    
    # 3th plot altitude vs intensity for all data
    graph_plot(all_intensity, all_altitude, 'intensity (SNR + 1)', 'Altitude (m)', 'Intensity vs altitude (for all data)')
    
    # 3th plot altitude vs intensity for all data
    graph_plot(filtered_intensity, int_altitude, 'intensity (SNR + 1)', 'Altitude (m)', 'Intensity vs altitude (for filtered data)')
    
    # 5th plot altitude vs SNR all data
    graph_plot(all_SNR, all_altitude, 'SNR', 'Altitude (m)', 'SNR vs altitude (for all data)')
    
    # 6th plot altitude vs SNR for first batch
    graph_plot(filtered_SNR, SNR_altitude, 'SNR', 'Altitude (m)', 'SNR vs altitude (for filtered data)')
    
    # 7th plotting a scatter color map of the backscatter with altitude and time
    plotting_colormap(float_dec_time, all_altitude, all_backscatter, 'Time(H)', 'Altitude(m)', 'Backscatter', 'backscatter color map')
    
    # 8th plotting a scatter color map of the backscatter with altitude and time (Filtered)
    plotting_colormap(beta_time, beta_altitude, filtered_backscatter, 'Time(H)', 'Altitude(m)', 'Backscatter', 'Filtered backscatter color map')
    
    # 9th plotting scatter color map for intensity
    plotting_colormap(float_dec_time, all_altitude, all_intensity, 'Time(H)', 'Altitude(m)', 'Intensity(SNR+1)', 'Intensity color map')
    
    # 10th plotting a scatter color map for intensity (Filtered)
    plotting_colormap(int_time, int_altitude, filtered_intensity, 'Time(H)', 'Altitude(m)', 'Intensity', 'Filtered intensity color map')
    
    # 11th plotting scatter color map of the SNR
    plotting_colormap(float_dec_time, all_altitude, all_SNR, 'Time(H)', 'Altitude(m)', 'SNR', 'SNR color map')
    
    # 12th plotting a scatter color map of the SNR (Filtered)
    plotting_colormap(SNR_time, SNR_altitude, filtered_SNR, 'Time(H)', 'Altitude(m)', 'SNR', 'Filtered SNR color map')
    
    # 13th plotting scatter color map of the Doppler 
    plotting_colormap(float_dec_time, all_altitude, all_doppler, 'Time(H)', 'Altitude(m)', 'Doppler (m/s)', 'Doppler velocity color map')
    
    # 14th plotting scatter color map of the Doppler 
    plotting_colormap(dop_time, dop_altitude, filtered_doppler, 'Time(H)', 'Altitude(m)', 'Doppler (m/s)', 'filtered Doppler velocity color map')
    
 
def metadata_filter(data, index_list):
    """ Take the index data and return a list of filtered data"""
    filter_data = []
    for i in index_list:
        filter_data.append(data[i])
    return filter_data
        
def data_filter(intensity, upper, lower):
    """Take a list of intensity and we want to return the index list and the filter list"""
    filter_intensity = []
    filter_index = []
    for index, item in enumerate(intensity):
        if lower <= item <= upper:
            filter_intensity.append(item)
            filter_index.append(index)
    return filter_intensity, filter_index

    
def plotting_colormap(x, y, data_point, xlabel, ylabel, cbar_label, title):
    """Function that construct a color map graph"""
    # RdYlBu_r ()
    # viridis
    # hsv
    cmap = plt.get_cmap('Spectral_r')
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=data_point, cmap=cmap, marker='o')
    
    # Adding color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label(cbar_label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Rotate x-axis labels for better readability (optional)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def decimal_hours_to_hh_mm(decimal_hours):
    """Function to convert decimal hours to HH:MM format"""
    hours = int(decimal_hours)
    minutes = int((decimal_hours - hours) * 60)
    hours %= 24  # Ensure hours are within the 24-hour range
    return f"{hours:02d}:{minutes:02d}"
    
def extracting_data(i, data):
    """This function take the index of data that we want and return a list of that data"""
    data_list = []
    for item in data:
        specific_data = [float(sublist[i]) for sublist in item]
        data_list.append(specific_data)
    return data_list
    
def graph_plot(x, y, xlabel, ylabel, title):
    """plot a simple graph from the data"""
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def get_altitude(range_gate, gate_length):
    """Take the data from line 2 and compute the altitude using the formula given in the data"""
    all_altitude = []
    
        
    for item in range_gate:
        set_altitude = []
        for gate in item:
            altitude = gate_length/2 + (gate * 3)
            set_altitude.append(altitude)
        
        all_altitude.append(set_altitude)
        
    return all_altitude

def get_stats(data):
    """Computing the average, variance and stanard deviation of the data"""
    data_float = [float(x) for x in data]
    data_mean = np.mean(data_float)
    data_variance = np.var(data_float)
    data_stdev = np.std(data_float)
    return data_mean, data_variance, data_stdev
    
def read_file(path):
    """Function that take file path and return the list and number of file"""
    files = glob.glob(path + '**/*', recursive=False)
    n = len(files)
    return files, n

def read_metadata(files):
    """Function wil take a list of file iterate them and take specific data"""
    system_id = []          # 0
    num_of_gates = []       # 1
    range_gate_length = []  # 2
    gate_length = []        # 3
    pulse_per_ray = []      # 4
    focus_range = []        # 5
    start_time = []         # 6
    
    for file in files:
        with open(file, 'r') as input_file:  # Use 'with' to automatically close the file
            for line in input_file:
                if line.startswith('System ID:'):
                    system_id.append(line[11:].strip())
                elif line.startswith('Number of gates:'):
                    num_of_gates.append(line[17:].strip())
                elif line.startswith('Range gate length (m):'):
                    range_gate_length.append(line[23:].strip())
                elif line.startswith('Gate length (pts):'):
                    gate_length.append(line[19:].strip())
                elif line.startswith('Pulses/ray:'):
                    pulse_per_ray.append(line[12:].strip())
                elif line.startswith('Focus range:'):
                    focus_range.append(line[13:].strip())
                elif line.startswith('Start time:'):
                    start_time.append(line[12:].strip())

    metadata_array = np.array([system_id, num_of_gates, range_gate_length, gate_length, pulse_per_ray, focus_range, start_time], dtype=object)
    
    return metadata_array
    
def read_data(files):
    """Function takes a list of data, iterate them and take the numerical data"""
    
    # Line 1 are Decimal time(hours)  Azimuth(degrees)  Elevation(degrees) Pitch(degrees) Roll(degrees)
    # Line 2 are Range Gate  Doppler (m/s)  Intensity (SNR + 1)  Beta (m-1 sr-1)
    line1_list = []
    line2_list = []

    rows_per_block_line2 = 1801
    start_skiprows_line1 = 17
    start_skiprows_line2 = 18

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()

            for i in range(start_skiprows_line2, len(lines), rows_per_block_line2):
                line1_data = lines[i - 1].split()  # Split line1 data
                line2_data = [line.split() for line in lines[i:i + 1800]]  # Split line2 data

                line1_list.append(line1_data)
                line2_list.append(line2_data)
    
    return line1_list, line2_list
           
    
main()
