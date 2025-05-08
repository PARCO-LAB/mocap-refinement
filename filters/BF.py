import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
from timeit import default_timer as timer
import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# Sometimes is necessary to have a pre processing step (before runtime)
def pre_process(lowcut,fs,order):
    b, a = butter(N=order, Wn=lowcut, btype='low', analog=False, fs=fs)
    return b,a
    
# Called at each time-step:
# - skeleton is a list of values containing the coordinates [X,Y,Z] of each joint in row ( e.g. [1 5 2 6 7 1 ...] ).
# - time is a float expressed in seconds
# - history are all the past values of skeleton, so it's a table
def BF(skeleton, lowcut, fs, order, b,a):
    out = []
    for i in range(len(skeleton[0])):
        col = [p[i] for p in skeleton]
        y = filtfilt(b, a, col)
        out.append(y)

    new_out = np.array(out).T.tolist()
   
   
    return new_out

# Perform some operations at the end of the time frames
def post_process():
    pass


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def routine(table, delta):
    
    # Pre-process phase
    start_pre = timer()
    # ------------------------------------------------------------------------------------------------------------------
    out = table
    lowcut = 2
    order = 3
    b,a = pre_process(lowcut,30,order)
    
    # ------------------------------------------------------------------------------------------------------------------
    end_pre = timer()


    # Runtime phase
    start_run = timer()
    # ------------------------------------------------------------------------------------------------------------------
    # for i in range(0,(table.shape[0] - table.shape[0] % delta), delta):
    #     out.iloc[i:i+delta] = BF((table.iloc[i:i+delta].values), lowcut, 30, order,b,a)
    # out.iloc[table.shape[0]-table.shape[0]%delta:table.shape[0]] = table.iloc[table.shape[0]-table.shape[0] % delta:table.shape[0]]
    
    for i in range(0,table.shape[0]):
        if i < delta:
            pass
        else:
            out.iloc[i-delta+1:i+1] = np.array(BF((table.iloc[i-delta+1:i+1].values), lowcut, 30, order,b,a))
            # out.iloc[i,:] = seq[-1,:]
    
    # for i in range(0,(table.shape[0] - table.shape[0] % delta), delta):
    #     out.iloc[i:i+delta] = BF((table.iloc[i:i+delta].values), lowcut, 30, order,b,a)
    # out.iloc[table.shape[0]-table.shape[0]%delta:table.shape[0]] = table.iloc[table.shape[0]-table.shape[0] % delta:table.shape[0]]
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