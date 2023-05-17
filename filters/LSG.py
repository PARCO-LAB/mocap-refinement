import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
import viewer
import sys
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp

import warnings
import scipy.signal as signal
import torch
from scipy.ndimage.filters import gaussian_filter1d
import time

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
def LS(skeleton):
    out = []
    for i in range(len(skeleton[0])):
        col = [p[i] for p in skeleton]
        n = len(col)
        mean = sum(col) / n
        var = sum((x - mean)**2 for x in col) / n
        std_dev = var ** 0.5
        y = gaussian_filter1d(col, 100*std_dev)
        out.append(y)

    new_out = np.array(out).T.tolist()
    
    return new_out      

# Perform some operations at the end of the time frames
def post_process():
    pass


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def routine(table, time, names):
    
    out = []
    
    # Pre-process phase
    start_pre = timer()
    # ------------------------------------------------------------------------------------------------------------------
    out = table
    
    # ------------------------------------------------------------------------------------------------------------------
    end_pre = timer()

    # Runtime phase
    start_run = timer()
    # ------------------------------------------------------------------------------------------------------------------

    delta = 20
    
    for i in range(0,(len(time) - len(time) % delta), delta):
        out[i:i+delta] = LS((table[i:i+delta]))
    #out[len(time)-len(time)%delta:len(time)] = LS((table[len(time)-len(time)%delta:len(time)]))
    out[len(time)-len(time)%delta:len(time)] = table[len(time)-len(time)%delta:len(time)]
    
    print(len(table), len(out))
    
    
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
    table_out = routine(table, time, names)
    # ------------------------------------------------------------------------------------------------------------------
    #output_path = input_path.replace("input","output/"+filter_name)
    output_path = f+"/"+file_name
    viewer.write_table(output_path,table_out, time, names)


if __name__ == "__main__":
    main()