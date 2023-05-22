from math import nan,isnan
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
import viewer
from timeit import default_timer as timer
import numpy as np
from scipy import signal
import pandas as pd
from scipy.interpolate import interp1d
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# Sometimes is necessary to have a pre processing step (before runtime)
def pre_process():
    pass

# Called at each time-step:
# - skeleton is a list of values containing the coordinates [X,Y,Z] of each joint in row ( e.g. [1 5 2 6 7 1 ...] ).
# - time is a float expressed in seconds
# - history are all the past values of skeleton, so it's a table
def INT(matrix):
    # Determine the number of rows and columns in the matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Create an output matrix with the same size as the input matrix
    interpolated_matrix = [[None] * num_cols for _ in range(num_rows)]

    # Iterate over each column in the matrix
    for col in range(num_cols):
        # Extract the column values
        column_values = [row[col] for row in matrix]

        # Find the indices of None values in the column
        none_indices = [i for i, value in enumerate(column_values) if isnan(value)]
        # Find the indices of non-None values in the column
        non_none_indices = [i for i in range(num_rows) if i not in none_indices]
        #print(column_values)
        # Perform spline interpolation for None values
        f = interp1d(non_none_indices, [column_values[i] for i in non_none_indices], kind='cubic')
        interpolated_values = [f(i) if i in none_indices else column_values[i] for i in range(num_rows)]

        # Update the corresponding column in the interpolated matrix
        for row in range(num_rows):
            interpolated_matrix[row][col] = interpolated_values[row]

    return interpolated_matrix

# Perform some operations at the end of the time frames
def post_process():
    pass


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def routine(table, time, names, delta):

    out = []

    # Pre-process phase
    start_pre = timer()
    # ------------------------------------------------------------------------------------------------------------------

    pre_process()

    # ------------------------------------------------------------------------------------------------------------------
    end_pre = timer()

    # Runtime phase
    start_run = timer()
    # ------------------------------------------------------------------------------------------------------------------
    
    
    out = INT(table)
    
    # ------------------------------------------------------------------------------------------------------------------
    end_run = timer()

    # Post processing phase
    start_post = timer()
    # ------------------------------------------------------------------------------------------------------------------

    post_process()

    # ------------------------------------------------------------------------------------------------------------------
    end_post = timer()

    kps_num = int((len(names)-1)/3)
    pre_time = round(end_pre-start_pre,5)*1000
    run_time = round(end_run-start_run,5)*1000
    post_time = round(end_post-start_post,5)*1000
    tot_time = round((pre_time+run_time+post_time),2)
    print("INFO:\tkps:",kps_num,"\tframes:",len(out),"\tdelay:", round(tot_time/len(out),3) ,"ms")    
    print("TIME ELAPSED:\tpre:",round(end_pre-start_pre,5)*1000,"ms\trun:",round(end_run-start_run,5)*1000,"ms\tpost:",round(end_post-start_post,5)*1000,"ms")
    
    return out

# Parse argument if passed directly from viewer.py
def main():
    global input_path, filter_name
    fs = float(sys.argv[2])
    delta = 2*int(sys.argv[3])
    filter_name = sys.argv[0].split('/')[-1].replace('.py','')
    input_path =sys.argv[1]
    f = input_path.replace("input","output").replace(input_path.split('/')[-1],'')+filter_name
    file_name =  sys.argv[1].split('/')[-1]
    if not os.path.isdir(f):
      os.makedirs(f)
    table, time, names = viewer.get_table(input_path)
    # ------------------------------------------------------------------------------------------------------------------
    table_out = routine(table, time, names,delta)
    # ------------------------------------------------------------------------------------------------------------------
    #output_path = input_path.replace("input","output/"+filter_name)
    output_path = f+"/"+file_name
    viewer.write_table(output_path,table_out, time, names)

if __name__ == "__main__":
    main()