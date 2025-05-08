from math import nan,isnan
import pandas as pd
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
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
        f = interp1d(non_none_indices, [column_values[i] for i in non_none_indices], kind='cubic', fill_value="extrapolate")
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
def routine(table, delta):
    delta += 1
    out = table.copy(deep = True)

    # Pre-process phase
    start_pre = timer()
    # ------------------------------------------------------------------------------------------------------------------

    pre_process()

    # ------------------------------------------------------------------------------------------------------------------
    end_pre = timer()

    # Runtime phase
    start_run = timer()
    # ------------------------------------------------------------------------------------------------------------------
    
    # out.iloc[:,:] = INT(table.iloc[:,:].values)
    
    # for i in range(0,(table.shape[0] - table.shape[0] % delta), delta):
    #     out.iloc[i:i+delta,:] = INT(table.iloc[i:i+delta,:].values)
    # if table.shape[0] % delta != 0:
    #     out.iloc[table.shape[0]-delta:table.shape[0],:] = INT(table.iloc[table.shape[0]-delta:table.shape[0],:].values)
    
    out.iloc[0:delta,:] = np.array(INT((table.iloc[0:delta].values)))
    for i in range(0,table.shape[0]):
        if i < delta:
            pass
        else:
            out.iloc[i,:] = np.array(INT((table.iloc[i-delta+1:i+1].values)))[-1,:]
    
    # ------------------------------------------------------------------------------------------------------------------
    end_run = timer()

    # Post processing phase
    start_post = timer()
    # ------------------------------------------------------------------------------------------------------------------

    post_process()

    # ------------------------------------------------------------------------------------------------------------------
    end_post = timer()

    kps_num = int(table.shape[1]/3)
    pre_time = round(end_pre-start_pre,5)*1000
    run_time = round(end_run-start_run,5)*1000
    post_time = round(end_post-start_post,5)*1000
    tot_time = round((pre_time+run_time+post_time),2)
    print("INFO:\tkps:",kps_num,"\tframes:",len(out),"\tdelay:", round(tot_time/len(out),3) ,"ms")    
    print("TIME ELAPSED:\tpre:",round(end_pre-start_pre,5)*1000,"ms\trun:",round(end_run-start_run,5)*1000,"ms\tpost:",round(end_post-start_post,5)*1000,"ms")
    return out

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('filter_name')
parser.add_argument('input_path')
parser.add_argument('output_path')
parser.add_argument('delta')

args = parser.parse_args()

# Parse argument if passed directly from viewer.py
def main():
    delta = int(args.delta)
    filter_name = args.filter_name
    input_path = args.input_path
    f = args.output_path + filter_name
    file_name =  input_path.split('/')[-1]
    if not os.path.isdir(f):
      os.makedirs(f)
    
    # ------------------------------------------------------------------------------------------------------------------
    table_out = routine(pd.read_csv(input_path),delta)
    # ------------------------------------------------------------------------------------------------------------------
    output_path = os.path.join(f,file_name)
    table_out.to_csv(output_path)

if __name__ == "__main__":
    main()