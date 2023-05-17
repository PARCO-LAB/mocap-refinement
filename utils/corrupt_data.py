from operator import delitem
import pandas as pd
import argparse
import os
import numpy as np

def add_gaussian_noise(tab,noise):
    len = tab.shape[0]
    for (index, column) in enumerate(tab):
        if index > 0:
            tab[column] += np.random.normal(0, 0.1, len)
    return tab

def main(args):
    src_name = args.src[0]
    out_name = args.out[0]

    for filename in os.listdir(src_name):    
        f = os.path.join(src_name, filename)
        if os.path.isfile(f):    
            src = pd.read_csv(f, delimiter=',')
            
            if args.mode[0] == "denoising" :
                
                dst = add_gaussian_noise(src,float(args.gaussian[0]))
                
            elif args.mode[0] == "completion":
                pass
            dst.to_csv(os.path.join(out_name, filename),index=False)

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
