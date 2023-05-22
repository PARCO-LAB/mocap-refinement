from operator import delitem
import random
import pandas as pd
import argparse
import os
import numpy as np
keypoints = ['RShoulder', 'LAnkle', 'LFoot', 'Head', 'LShoulder', 'RWrist', 'RFoot', 'RHip', 'LHip', 'RAnkle', 'Neck', 'RElbow', 'LKnee', 'LElbow', 'LWrist', 'RKnee']
def add_gaussian_noise(tab):
    len = tab.shape[0]
    for (index, column) in enumerate(tab):
        if index > 0:
            tab[column] += np.random.normal(0, 0.1, len)
    return tab

def add_gaps(tab):
    # Copia il dataframe in input per evitare modifiche indesiderate
    tab_copy = tab.copy()
    
    # Ottiene il numero di frame e il numero di keypoints
    n_frames = tab_copy.shape[0]
    n_keypoints = tab_copy.shape[1] // 3
    
    # Genera indici casuali per le coordinate dei keypoints
    random_frames = np.random.choice(np.arange(1,n_frames-1), size=(round(n_frames/3)), replace=True)
    random_indices = np.random.choice(np.arange(n_keypoints), size=(round(n_frames/3), round(n_keypoints/3)), replace=True)
    
    # Inserisce valori NaN casuali nelle coordinate dei keypoints
    for i in range(len(random_frames)):
        for j in random_indices[i]:
            keypoint_col = keypoints[j]
            for coord in ["X", "Y", "Z"]:
                col = f"{keypoint_col}:{coord}"
                tab_copy.at[random_frames[i], col] = np.nan    
    return tab_copy

def main(args):
    src_name = args.src[0]
    out_name = args.out[0]

    for filename in os.listdir(src_name):    
        f = os.path.join(src_name, filename)
        if os.path.isfile(f):    
            src = pd.read_csv(f, delimiter=',')
            
            if args.mode[0] == "denoising" :
                
                dst = add_gaussian_noise(src)
            elif args.mode[0] == "completion":
                dst = add_gaps(src)

            dst.to_csv(os.path.join(out_name, filename),index=False,na_rep='NaN')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the data for being corrupted", epilog="PARCOLAB")
    parser.add_argument("--source",
                        "-s",
                        dest="src",
                        required=True,
                        nargs=1,
                        help="path to the source keypoints file")
    parser.add_argument("--type",
                        "-t",
                        dest="mode",
                        required=True,
                        nargs=1,
                        help="\{denoising,completion\}")
    parser.add_argument("--out",
                        "-o",
                        dest="out",
                        required=True,
                        nargs=1,
                        help="path of output files")
    parser.add_argument("--gaussian_noise",
                        "-g",
                        dest="gaussian",
                        required=False,
                        nargs=1,
                        default='0',    
                        help="")
    args = parser.parse_args()
    main(args)
