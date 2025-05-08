import sys,os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
from timeit import default_timer as timer
import numpy as np
import random
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def compute_distance(a,b):
    return np.sqrt( np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2)+np.power(a[2]-b[2],2) )

def compute_velocity(now,old,dt):
    vel = []
    for j in range(0,len(now),3):
        dist = abs(compute_distance(now[j:j+3],old[j:j+3]))
        vel.append(float(dist)/dt)
    return vel

def weight_average(kf,de,vel,std):
    finale = []
    for i in range(len(kf)):
        if vel[i]/std[i] < 1:
            K = 0
        elif vel[i]/std[i] > 100:
            K = 0.9
        else:
            K = -2.5*(vel[i]/std[i])**(-0.09691) + 2.5
        finale.append( (1-K)*kf[i] + K*de[i] )
    return finale

def get_bone_length(s,names):
    out_bones_length = []
    out_names = []
    bones = {
    "Spine" : ["MidShoulder","MidHip"],
    "RHumerus" : ["RShoulder","RElbow"],
    "LHumerus" : ["LShoulder","LElbow"],
    "RForearm" : ["RWrist","RElbow"],
    "LForearm" : ["LWrist","LElbow"],
    "RFemur" : ["RHip","RKnee"],
    "LFemur" : ["LHip","LKnee"],
    "RTibia" : ["RAnkle","RKnee"],
    "LTibia" : ["LAnkle","LKnee"],
    "RFoot" : ["RAnkle","RFoot"],
    "LFoot" : ["LAnkle","LFoot"],
    }

    for b in bones:
        try:
            i1 = [idx for idx, l in enumerate(names) if bones[b][0] in l][0]
            i2 = [idx for idx, l in enumerate(names) if bones[b][1] in l][0]
            out_bones_length.append(compute_distance(s[i1-1:i1+2],s[i2-1:i2+2]))
            out_names.append(b)
        except:
            pass
    return out_bones_length,out_names        
            


class DE():
    def __init__(self,pop_size=10,iter_num=10,mutation_factor=1,crossover_rate=0.7):
        self.pop_size = pop_size
        self.gt_bl = []
        self.names = []
        self.in_names = []
        self.iterations = iter_num
        self.mf = mutation_factor
        self.cr = crossover_rate
    # save std dev for the initial population generation
    def compute_std_dev(self,data):
        self.std_dev = [np.std(data[:,d]) for d in range(data.shape[1])]
    
    def update_bounds(self,s):
        bounds = []
        for i in range(len(s)):
            if s[i][1] >= 0:
                bounds.append( [ float(s[i][0]) , float(s[i][0]+self.std_dev[i]) ] )
            else:
                bounds.append( [ float(s[i][0]-self.std_dev[i]) , float(s[i][0]) ] )
        self.bounds = np.array(bounds)

    def initialize_population(self):
        self.population = self.bounds[:, 0] + (np.random.rand(self.pop_size, len(self.bounds)) * (self.bounds[:, 1] - self.bounds[:, 0]))

    def initialize_bone_lengths(self,skeletons,names):
        
        
        self.in_names = names

        # Dinamically update
        bones = []
        for s in skeletons:
            bl,self.names = get_bone_length(s,names)
            bones.append(bl)
        self.gt_bl = np.mean(np.array(bones), axis=0)
        
        
        # Statically get the ground truth bones length -> "the oracle"
        """
        bones = []
        gt_path = input_path.replace("input","ground_truth")
        table, time, gt_names = viewer.get_table(gt_path)
        for row in table:
            bl,static_names = get_bone_length(row,gt_names)
            bones.append(bl)
        
        static_bl = np.mean(np.array(bones), axis=0)
        
        # Check if there are all the possible bones
        for i in range(len(self.names)):
            if self.names[i] in static_names:
                self.gt_bl[i] = static_bl[static_names.index(self.names[i])]

        """
        

    def obj(self,x):
        bl,names = get_bone_length(x,self.in_names)
        return np.sum( abs(np.array(self.gt_bl) - np.array(bl)) )
        
    def evaluate(self):
        #print(self.population.shape)
        res = [self.obj(ind) for ind in self.population]
        return res

    def mutation(self,x):
        return x[0] + self.mf*(x[1]-x[2])

    def check_bounds(self,x):
        y = [np.clip(x[i], self.bounds[i, 0], self.bounds[i, 1]) for i in range(len(self.bounds))]
        return y

    def crossover(self,mutated,target):
        p = np.random.rand(self.bounds.shape[0])
        return [mutated[i] if p[i] < self.cr else target[i] for i in range(self.bounds.shape[0])]


    def find(self,states):
        self.update_bounds(states)    
        self.initialize_population()
        objs = self.evaluate()
        best_obj = min(objs)
        prev_obj = best_obj
        best_vector = self.population[np.argmin(objs)]
        # iterations
        for i in range(self.iterations):
            for j in range(self.pop_size):
                # Mutation
                candidates = [candidate for candidate in range(self.pop_size) if candidate != j]
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                mutated = self.check_bounds(self.mutation([a, b, c]))
                trial = self.crossover(mutated,self.population[j])
                trial_score = self.obj(trial)
                if trial_score < objs[j]:
                    self.population[j] = trial
                    objs[j] = trial_score
            best_obj = min(objs)
            if best_obj < prev_obj:
                best_vector = self.population[np.argmin(objs)]
                prev_obj = best_obj
                if abs( 1/(best_obj+1)-1/(prev_obj+1) ) < 0.001:
                    return best_vector
        return best_vector

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
        return self.Y
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
        kalman[i].update(sk[i],[1])
        res.append(kalman[i].get_output())
    return res

# Perform some operations at the end of the time frames
def post_process():
    pass


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def routine(table,delta,names):
    fs = 30
    out = table.copy(deep=True)
    
    # Pre-process phase
    start_pre = timer()
    # ------------------------------------------------------------------------------------------------------------------
    tab = table.iloc[:,:].values
    kalman = pre_process( tab.shape[1],fs, tab[0,:] )
    de = DE(iter_num=100,pop_size=100,mutation_factor=2,crossover_rate=0.5)
    # ------------------------------------------------------------------------------------------------------------------
    end_pre = timer()

    # Runtime phase
    start_run = timer()
    # ------------------------------------------------------------------------------------------------------------------
    epsilon = 1
    
    std_dev_computed = False

    for i in range(1,table.shape[0]):
        # Compute velocity for each joint
        vel = compute_velocity(tab[i,:],tab[i-1,:],1/fs)
        y = kalman_filter(kalman, tab[i,:])
        # If all the velocities are below the treshold, keep going with KF only
        if np.all(np.array([v < epsilon for v in vel])) or i < delta:
            out.iloc[i,:] = y
        # Use DE otherwise
        else:
            # Calculate the standard deviation
            if not std_dev_computed:
                std_dev_computed = True
                de.compute_std_dev(tab[0:delta,:])
                de.initialize_bone_lengths(tab[0:delta,:],names)
            res = de.find([k.X for k in kalman])
            weighted = weight_average(y,res,[float(k.X[1]) for k in kalman],de.std_dev)
            
            # Output the averaged or only the DE?
            # - Das2017 choice
            #out.append(res)
            # - My personal choice
            score_kf = de.obj(y)
            score_de = de.obj(res)
            score_we = de.obj(weighted)
            results = [y,res,weighted]
            out.iloc[i,:] = results[np.argmin([score_kf,score_de,score_we])]
            

            # Update kalman state
            for j in range(len(kalman)):
                #- Das2017 choice
                kalman[j].X[0] = weighted[j]
                # Mine
                #kalman[j].X[0] = results[np.argmin([score_kf,score_de,score_we])][j]
            
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
    
    table = pd.read_csv(input_path)
    names = [ el.split(":")[0] for el in list(table.columns)]
    names = [names[i] for i in range(len(names)) if i % 3 == 0]
        
    # ------------------------------------------------------------------------------------------------------------------
    table_out = routine(table,delta,names)
    # ------------------------------------------------------------------------------------------------------------------
    output_path = os.path.join(f,file_name)
    table_out.to_csv(output_path)

if __name__ == "__main__":
    main()