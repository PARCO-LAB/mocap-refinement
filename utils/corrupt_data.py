import pandas as pd
import argparse

def main(args):
    src_name = args.src[0]
    out_name = args.out[0]
    
    src = pd.read_csv(src_name, delimiter=',')
    
    if args.mode == "denoising" :
        dst = add_gaussian_noise(src)
    elif args.mode == "completion":
        dst = add_gaps(src)

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
                        "-c",
                        dest="side",
                        required=False,
                        nargs=1,
                        default='15',
                        help="")
    args = parser.parse_args()
    main(args)