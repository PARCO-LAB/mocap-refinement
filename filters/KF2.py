import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
import viewer
from timeit import default_timer as timer
import numpy as np
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def gauss_pdf(X,M,S):
    if M.shape[1] == 1:
        DX = X - np.tile(M, X.shape[1])
        E = 0.5 * sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(det(S))
        P = np.exp(-E)
    elif X.shape[1] == 1:
        DX = np.tile(X, M.shape[1])- M
        E = 0.5 * sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.det(S))
        P = np.exp(-E)
    else:
        DX = X-M
        E = 0.5 * np.dot(DX.T, np.dot(np.linalg.inv(S), DX))
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.det(S))
        P = np.exp(-E)
    return P[0], E[0]


class Kalman():
    def __init__(self,fs,s):
        dt = 1/fs
        self.X = np.array([[s],[0.1],[0.01]])
        self.P = np.diag((1, 1, 1))
        self.F = np.array([[1, dt, dt*dt/2], [0, 1, dt], [0, 0, 1]])
        self.Q = np.eye(self.X.shape[0])*0.05
        self.Y = np.array([s])
        self.H = np.array([1, 0, 0]).reshape(1,3)
        self.R = 1 # np.eye(self.Y.shape[0])*
        
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
def routine(table, time, names,fs):
    
    out = []
    
    # Pre-process phase
    start_pre = timer()
    # ------------------------------------------------------------------------------------------------------------------
    tab = np.array(table)
    kalman = pre_process( tab.shape[1],fs, table[0] )
    
    # ------------------------------------------------------------------------------------------------------------------
    end_pre = timer()

    # Runtime phase
    start_run = timer()
    # ------------------------------------------------------------------------------------------------------------------

    out.append(table[0])
    for i in range(1,len(time)):
        out.append(kalman_filter(kalman,tab[i,:]))


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
    table_out = routine(table, time, names,fs)
    # ------------------------------------------------------------------------------------------------------------------
    #output_path = input_path.replace("input","output/"+filter_name)
    output_path = f+"/"+file_name
    viewer.write_table(output_path,table_out, time, names)

if __name__ == "__main__":
    main()