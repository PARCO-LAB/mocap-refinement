import sys,os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
from timeit import default_timer as timer
import numpy as np
import scipy as sp
import statistics
from fancyimpute import SoftImpute

from rpca import ialm 

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

def RPCA(data):
    """
    Perform Robust PCA to separate low-rank structure and sparse noise.

    Parameters:
        data (np.ndarray or pd.DataFrame): Input matrix (can have missing or noisy entries).
        return_sparse (bool): If True, also return the sparse matrix of outliers.

    Returns:
        np.ndarray: Low-rank matrix (denoised/reconstructed).
        (optional) np.ndarray: Sparse matrix (outliers/noise).
    """
    # Convert DataFrame to numpy array if necessary
    if hasattr(data, "values"):
        data = data.values

    # Replace NaNs with zeros; alternatively, consider imputing missing values beforehand
    data = np.nan_to_num(data)

    # Perform Robust PCA using the Inexact ALM algorithm
    L, S = ialm.fit(data,verbose=False)
    return L


# Perform some operations at the end of the time frames
def post_process():
    pass


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def routine(table, delta):
    
    out = table.copy(deep=True)
    
    # Pre-process phase
    start_pre = timer()
    # ------------------------------------------------------------------------------------------------------------------
    
    pre_process()
    
    # ------------------------------------------------------------------------------------------------------------------
    end_pre = timer()

    # Runtime phase
    start_run = timer()
    # ------------------------------------------------------------------------------------------------------------------
    
    
    for i in range(0,(table.shape[0] - table.shape[0] % delta), delta):
        out.iloc[i:i+delta,:] = RPCA(table.iloc[i:i+delta,:].values)
    if table.shape[0] % delta != 0:
        out.iloc[table.shape[0]-delta:table.shape[0],:] = RPCA(table.iloc[table.shape[0]-delta:table.shape[0],:].values)
    
    
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