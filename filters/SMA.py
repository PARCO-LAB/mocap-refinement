import viewer
import sys
from timeit import default_timer as timer

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
def SMA(skeleton):
    out = []
    for i in range(len(skeleton[1])):
        col = [p[i] for p in skeleton]
        out.append(sum(col)/len(col))
    return out

# Perform some operations at the end of the time frames
def post_process():
    pass

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def routine(table, time, names,delta):
    
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
    for i in range(0,len(time)):
        if i < delta/2 or i > len(time)-(delta/2):
            out.append(table[i])
        else:
            out.append(SMA(table[i-int(delta/2):i+int(delta/2)]))
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
    viewer.write_time(tot_time,pre_time,run_time,post_time,kps_num,len(out),input_path.replace("input","output/"+filter_name).replace(".csv","_time.csv"))

    return out

# Parse argument if passed directly from viewer.py
def main():
    global input_path, filter_name
    delta = 2*int(sys.argv[3])
    filter_name = sys.argv[0].split('/')[-1].replace('.py','')
    input_path = input_path=sys.argv[1]
    import os
    if not os.path.isdir(input_path.replace("input","output/"+filter_name).replace('.csv','')):
      os.makedirs(input_path.replace("input","output/"+filter_name).replace('.csv',''))
    table, time, names = viewer.get_table(input_path)
    # ------------------------------------------------------------------------------------------------------------------
    table_out = routine(table, time, names,delta)
    # ------------------------------------------------------------------------------------------------------------------
    output_path = input_path.replace("input","output/"+filter_name)
    viewer.write_table(output_path,table_out, time, names)
    

if __name__ == "__main__":
    main()