import argparse
import pandas as pd
import numpy as np
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
import json
import os

def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))

header_bvh = """HIERARCHY
ROOT MidHip
{
    OFFSET  0 0 0
    CHANNELS 3 Xposition Yposition Zposition
    JOINT LHip
    {
        OFFSET 0 0 0
        CHANNELS 3 Xposition Yposition Zposition
        JOINT LKnee
        {
            OFFSET 0 0 0
            CHANNELS 3 Xposition Yposition Zposition
            JOINT LAnkle
            {
                OFFSET 0 0 0
                CHANNELS 3 Xposition Yposition Zposition
            }
        }
    }
    JOINT RHip
    {
        OFFSET 0 0 0
        CHANNELS 3 Xposition Yposition Zposition
        JOINT RKnee
        {
            OFFSET 0 0 0
            CHANNELS 3 Xposition Yposition Zposition
            JOINT RAnkle
            {
                OFFSET 0 0 0
                CHANNELS 3 Xposition Yposition Zposition
            }
        }
    }
    JOINT MidShoulder
    {
        OFFSET 0 0 0
        CHANNELS 3 Xposition Yposition Zposition
        JOINT LShoulder
        {
            OFFSET 0 0 0
            CHANNELS 3 Xposition Yposition Zposition
            JOINT LElbow
            {
                OFFSET 0 0 0
                CHANNELS 3 Xposition Yposition Zposition
                JOINT LWrist
                {
                    OFFSET 0 0 0
                    CHANNELS 3 Xposition Yposition Zposition
                }
            }
        }
        JOINT RShoulder
        {
            OFFSET 0 0 0
            CHANNELS 3 Xposition Yposition Zposition
            JOINT RElbow
            {
                OFFSET 0 0 0
                CHANNELS 3 Xposition Yposition Zposition
                JOINT RWrist
                {
                    OFFSET 0 0 0
                    CHANNELS 3 Xposition Yposition Zposition
                }
            }
        }   
    }
}
MOTION
"""

# Each keypoint and its parent
kps = [
    ["MidHip",None],
    ["LHip","MidHip"],
    ["LKnee","LHip"],
    ["LAnkle","LKnee"],
    ["RHip","MidHip"],
    ["RKnee","RHip"],
    ["RAnkle","RKnee"],
    ["MidShoulder","MidHip"],
    ["LShoulder","MidShoulder"],
    ["LElbow","LShoulder"],
    ["LWrist","LElbow"],
    ["RShoulder","MidShoulder"],
    ["RElbow","RShoulder"],
    ["RWrist","RElbow"],
]

tree = {
    "MidHip": [
        {"LHip" : { "LKnee" : {"LAnkle" : {}}}},
        {"RHip" : { "RKnee" : {"RAnkle" : {}}}},
        {"MidShoulder" :
            [
        {"LShoulder" : { "LElbow" : {"LWrist" : {}}}},
        {"RShoulder" : { "RElbow" : {"RWrist" : {}}}}
            ]
        }
    ]    
}


def writebvh(table,filename,frequency=0.03,offset=[0,0,0]):
    out = header_bvh
    out +="Frames: " + str(table.shape[0])
    out += "\nFrame Time: "+str(0.03)+"\n"
    for i in range(table.shape[0]):
        row = []
        for kp, parent in kps:
            if kp == "MidHip":
                A = np.array([table["RHip:X"][i],table["RHip:Y"][i],table["RHip:Z"][i]])
                B = np.array([table["LHip:X"][i],table["LHip:Y"][i],table["LHip:Z"][i]])
                point = (A+B) / 2 + np.array(offset)
            elif kp == "MidShoulder":
                A = np.array([table["RShoulder:X"][i],table["RShoulder:Y"][i],table["RShoulder:Z"][i]])
                B = np.array([table["LShoulder:X"][i],table["LShoulder:Y"][i],table["LShoulder:Z"][i]])
                point = (A+B) / 2
            else:
                point = np.array([table[kp+":X"][i],table[kp+":Y"][i],table[kp+":Z"][i]])
            if parent:
                if parent == "MidHip":
                    A = np.array([table["RHip:X"][i],table["RHip:Y"][i],table["RHip:Z"][i]])
                    B = np.array([table["LHip:X"][i],table["LHip:Y"][i],table["LHip:Z"][i]])
                    p = (A+B) / 2
                elif parent == "MidShoulder":
                    A = np.array([table["RShoulder:X"][i],table["RShoulder:Y"][i],table["RShoulder:Z"][i]])
                    B = np.array([table["LShoulder:X"][i],table["LShoulder:Y"][i],table["LShoulder:Z"][i]])
                    p = (A+B) / 2
                else:
                    p = np.array([table[parent+":X"][i],table[parent+":Y"][i],table[parent+":Z"][i]])
                point -= p
            # Rotate point
            # M0 = M([1,0,0], np.pi/2)*M([0,1,0], np.pi)
            
            # M0 = np.array(
            #     [
            #         [-1, 0,  0],
            #         [0, -1,  0],
            #         [0,  0,  1]
            #     ]
            # )
            
            M0 = np.dot(np.array(
                [
                    [1,0,0],
                    [0,0,-1],
                    [0,1,0]
                ]
            ),np.array(
                [
                    [-1,0,0],
                    [0,1,0],
                    [0,0,-1]
                ]
            ))
        
            point = dot(M0,point)
            row += list(point)
        for r in row:
            out += str(round(r,4)) + "\t"
        out =  out[:-1] + '\n'
    bvh = open(filename, "w")
    bvh.write(out)
    bvh.close()

CONTINUOUS_STATE_PARTS = [
            "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", 
            "right_knee", "left_ankle", "right_ankle", "neck"]
mapping = [12, 7, 10, 4,  5, 9, 6, 8, 11, 3, 14, 13]

new_map = [CONTINUOUS_STATE_PARTS[m] for m in mapping]

header = []
for m in mapping:
    part = CONTINUOUS_STATE_PARTS[m].replace("left_","L").replace("right_","R").replace("elbow","Elbow").replace("wrist","Wrist").replace("shoulder","Shoulder").replace("hip","Hip").replace("knee","Knee").replace("ankle","Ankle")
    header += [part + ":X",part + ":Y",part + ":Z"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="input a path to csv file to convert", type=str)
    parser.add_argument("-o","--output", help="input a path dir to store the bvh files", type=str)
    parser.add_argument("-f","--frequency", help="input the acquired frequency", type=float, default=0.03)
    parser.add_argument("-x","--offset_x", help="offset in the x direction (m)", type=float, default=0.0)
    parser.add_argument("-y","--offset_y", help="offset in the y direction (m)", type=float, default=0.0)
    parser.add_argument("-z","--offset_z", help="offset in the z direction (m)", type=float, default=0.0)
    args = parser.parse_args()
    
    # From csv to table
    df = pd.read_csv(args.input)
    
    if args.output[-1] == "\\":
        outfilename = os.path.join(args.output,args.input.split("/")[-1].replace("csv","bvh"))
    else:
        outfilename = os.path.join(args.output+"_"+args.input.split("/")[-1].replace("csv","bvh"))
    
    writebvh(df,outfilename, args.frequency,[args.offset_x,args.offset_y,args.offset_z])