import sys,os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
from timeit import default_timer as timer
import numpy as np
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
class Kalman():
    def __init__(self,fs,s):
        dt = 1/fs
        self.X = np.array([[s]])
        self.P = np.array([[1]])
        self.F = np.array([[1]])
        self.Q = np.eye(self.X.shape[0])*0.05
        self.Y = np.array([s])
        self.H = np.array([1]).reshape(1,1)
        self.R = 1 
        
    def predict(self):
        self.X = np.dot(self.F,self.X) #+ np.dot(self.B,self.U)
        self.P = np.dot(self.F, np.dot(self.P,self.F.T)) + self.Q

    def update(self,Y,R):
        self.Y = Y
        self.R = R
        self.K = np.dot(self.P,self.H.T) / ( R + np.dot(self.H,np.dot(self.P,self.H.T)) ) 
        self.X = self.X + self.K * ( Y - np.dot(self.H,self.X))
        self.P = np.dot((np.eye(self.X.shape[0])- np.dot(self.K,self.H)),self.P)
        self.Y = float(np.dot(self.H,self.X))

    def get_output(self):
        return float(np.dot(self.H,self.X))
# ------------------------------------------------------------------------------------------------------------------

# Sometimes is necessary to have a pre processing step (before runtime)
def pre_process(num,fs,sk):
    return [Kalman(fs,sk[i]) for i in range(num)]

# Called at each time-step:
# - skeleton is a list of values containing the coordinates [X,Y,Z] of each joint in row ( e.g. [1 5 2 6 7 1 ...] ).
# - time is a float expressed in seconds
# - history are all the past values of skeleton, so it's a table
def kalman_filter(kalman,sk):
    res = []
    for i in range(len(kalman)):
        kalman[i].predict()
        if not np.isnan(sk[i]):
            kalman[i].update(sk[i],[1])
        res.append(kalman[i].get_output())
    return res

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
    tab = table.iloc[:,:].values
    
    kalman = pre_process( tab.shape[1],30, tab[0,:] )
    
    # ------------------------------------------------------------------------------------------------------------------
    end_pre = timer()

    # Runtime phase
    start_run = timer()
    # ------------------------------------------------------------------------------------------------------------------

    
    for i in range(1,table.shape[0]):
        out.iloc[i,:] = kalman_filter(kalman,tab[i,:])


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