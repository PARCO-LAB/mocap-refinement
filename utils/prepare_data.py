import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob
import os
from concurrent.futures import ThreadPoolExecutor
import argparse

#warnings.filterwarnings("ignore")

def process_file(filename, scaler, delta, sample=None,_type="3D"):
    scaler = StandardScaler()
    X = []
    y = []
    df_in = pd.read_csv(filename[0])
    df_gt = pd.read_csv(filename[1])

    for j in range(delta,len(df_in)+1):

        # If u are training on clean poses only
        if filename[0] == filename[1]:
            in_seq = df_gt.iloc[j-delta:j,:].to_numpy()
            in_seq = scaler.fit_transform(in_seq)
            in_seq = np.nan_to_num(in_seq,nan=1)
            X.append(in_seq)
            y.append(in_seq)
        else:
            # --- Full error ---
            in_seq = df_in.iloc[j-delta:j,:].to_numpy()
            gt_seq = df_gt.iloc[j-delta:j,:].to_numpy()
            in_seq = scaler.fit_transform(in_seq)
            gt_seq = scaler.transform(gt_seq)
            X.append(in_seq)
            y.append(gt_seq)
            
            # # Add every combination clean partial
            # for d in range(1,delta):
            #     in_seq = df_in.iloc[j-delta:j,:].to_numpy()
            #     in_seq[0:delta-d,:] = df_gt.iloc[j-delta:j-d,:].to_numpy() # Clean values in the error ones
            #     gt_seq = df_gt.iloc[j-delta:j,:].to_numpy()
            #     in_seq = scaler.fit_transform(in_seq)
            #     in_seq = np.nan_to_num(in_seq,nan=1)
            #     gt_seq = scaler.transform(gt_seq)
            #     X.append(in_seq)
            #     y.append(gt_seq)
    print(np.array(X).shape,np.array(y).shape)
    return X,y

def processing(files, sample=None,split=0.8, delta=20,type="3D",threads=1):
    final_X = []
    final_y = []

    # Create a ThreadPoolExecutor with a max_workers parameter to control concurrency
    for filename in files:
        X , y = process_file(filename, scaler, delta, sample,_type=type)

        if len(X) > sample:
            subset_index = np.random.choice(list(range(0, len(X))),size = sample, replace = False)
            X = [X[i] for i in subset_index]
            y = [y[i] for i in subset_index]
        final_X += X
        final_y += y

    subset_index = np.random.choice(list(range(0, len(final_X))),size = len(final_X), replace = False)
    final_X = [final_X[i] for i in subset_index]
    final_y = [final_y[i] for i in subset_index]
    X = np.array(final_X)
    y = np.array(final_y)

    if split < 1:
        split_index = int(len(X) * split)  # 80% training, 20% validation
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
    else:
        X_train, X_val = X, None
        y_train, y_val = y, None

    return X_train, X_val, y_train, y_val, scaler



def find_keypoints(data):
    names = [ el.split(":")[0] for el in list(data.columns)]
    names = [names[i] for i in range(len(names)) if i % 3 == 0]
    return names

def center_on_hip_first(df : pd.DataFrame, kp_center = 'Hip'):
    df.reset_index(inplace = True)
    return df

def main(args):
    folder_name = args.method
    out_npy = args.out_path
    delta = args.delta
    in_path = os.path.join(args.path,folder_name)
    _sample = args.size
    # exist or not.
    if not os.path.exists(out_npy):
        os.makedirs(out_npy)
    global scaler
    scaler = 1

    files = [ (f,f.replace(folder_name,'gt')) for f in sorted(glob.glob(in_path + '/*.csv'))]
    X_train, X_val, y_train, y_val, scaler =  processing(files, _sample,split=0.8,delta=delta)
    print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)
    if np.any(X_train):
        np.save( out_npy + 'X_train_'  + str(delta) + "_" + str(_sample)+  '.npy', X_train)
        np.save( out_npy + 'X_val_' + str(delta) + "_" + str(_sample)+  '.npy', X_val)
        np.save( out_npy + 'y_train_' + str(delta) + "_" + str(_sample)+  '.npy', y_train)
        np.save( out_npy + 'y_val_' + str(delta) + "_" + str(_sample)+  '.npy', y_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="input size (e.g. 100, 800)", type=int)
    parser.add_argument("--method", help="method (e.g. spin, pare)", type=str)
    parser.add_argument("--delta", help="input dimension (e.g. 1, 10, 20, 30)", type=int)
    parser.add_argument("--path", help="path (e.g. '/home/emartini/nas/MAEVE/HUMAN_MODEL/dataset/review/h36m/train/')", type=str)
    parser.add_argument("--out_path", help="out_path (e.g. '/home/emartini/nas/MAEVE/HUMAN_MODEL/dataset/review/h36m/train/')", type=str)
    args = parser.parse_args()
    main(args)
