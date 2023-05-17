from math import sqrt
import argparse
import numpy as np
import pandas as pd
from scipy import interpolate
import os
pd.options.mode.chained_assignment = None 

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

def write_results(out_dir, mean_mae, std_mean_mae, mae, std_mae, mean_rmse, mean_std_rmse, rmse, pcc, keypoints):
    pcc.to_csv(out_dir + '_pcc.csv')

# out_dir, mean_tot,std_tot, mean_tot_r, std_tot_r, mean_tot_l, std_tot_l
def write_overall_results(out_dir, mean_mae, std_mean_mae, mean_kps, std_kps, pcc_mean, pcc_std, mpjpe,pampjpe, accel):
    print(os.path.basename(out_dir), "\t\tMPJPE:",round(mpjpe*1000,3), "\tPA-MPJPE:",round(pampjpe*1000,3),"\tAccel:",round(accel*1000,3))
    df = pd.DataFrame([[mean_mae, std_mean_mae, pcc_mean, pcc_std, mean_kps.max(), mean_kps.min(), mpjpe,pampjpe, accel ]],  columns=['MAE:AVG', 'MAE:STD', 'PCC:MAE', 'PCC:STD', 'MAE:MAX','MAE:MIN','MPJPE','PAMPJPE', 'Accel'])
    df.to_csv(out_dir + '_overall.csv')

def pearson_cross_correlation_coordinates(ref, src):
    all_kp = find_common_keypoints_full(ref, src)
    ref = ref.reindex(sorted(ref.columns), axis=1)
    src = src.reindex(sorted(src.columns), axis=1)
    ref = ref[all_kp]
    src = src[all_kp]
    pcc = ref.corrwith(src, axis = 0, method = 'pearson').to_frame()
    pcc.columns = ['PCC']
    return pcc

def root_mean_square_error(df):
    df.drop('time',axis=1, inplace=True, errors='ignore')
    rmse_df = pd.DataFrame()
    mean_df = df.mean()
    df = pd.DataFrame(columns = df.columns)

    for column in df:
        rmse_df[column] = [np.sqrt(mean_df[column])]

    mean_df_rmse = rmse_df.mean()

    return mean_df_rmse.mean(), mean_df_rmse.std(), rmse_df
    
def quadratic_distance_df(ref, src, keypoints):
    absolute_error = pd.DataFrame(src.time) 
    distance_list = []
    for kp in keypoints:
        p0 = []
        p1 = []
        dist = []
        for i in range(len(ref)):
            p0 = [ref[kp+':X'].loc[i],ref[kp+':Y'].loc[i],ref[kp+':Z'].loc[i]]
            p1 = [src[kp+':X'].loc[i],src[kp+':Y'].loc[i],src[kp+':Z'].loc[i]]
            dist.append(quadratic_distance(p0,p1))
        absolute_error[kp] = dist
    return absolute_error 

def quadratic_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2


def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c

def compute_error_accel(ref, src, kps, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    N = ref.shape[0]
    joints_gt = np.zeros(shape=(N, len(kps),3))
    joints_pred = np.zeros(shape=(N, len(kps),3))
    
    for i in range(len(kps)):
        joints_gt[:,i,0] = ref[kps[i]+':X'].values
        joints_gt[:,i,1] = ref[kps[i]+':Y'].values
        joints_gt[:,i,2] = ref[kps[i]+':Z'].values
        joints_pred[:,i,0] = src[kps[i]+':X'].values
        joints_pred[:,i,1] = src[kps[i]+':Y'].values
        joints_pred[:,i,2] = src[kps[i]+':Z'].values
    
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    #normed = np.norm(accel_pred - accel_gt, dim=2)
    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    acc=np.mean(normed[new_vis], axis=1)
    return np.mean(acc) #[~acc.isnan()]

def calculate_mpjpe(ref, src,kps):
    # Build the N*|kps|*3 matrices
    N = ref.shape[0]
    R = np.zeros(shape=(N, len(kps),3))
    S = np.zeros(shape=(N, len(kps),3))
    
    for i in range(len(kps)):
        R[:,i,0] = ref[kps[i]+':X'].values
        R[:,i,1] = ref[kps[i]+':Y'].values
        R[:,i,2] = ref[kps[i]+':Z'].values
        S[:,i,0] = src[kps[i]+':X'].values
        S[:,i,1] = src[kps[i]+':Y'].values
        S[:,i,2] = src[kps[i]+':Z'].values
    
    mpjpe = np.mean(np.sqrt(np.sum(np.square(S-R), axis=2)))

    pampjpe = np.zeros([N, len(kps)])

    for n in range(N):
        frame_pred = S[n]
        frame_gt = R[n]
        _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred = (b * frame_pred.dot(T)) + c
        pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))

    pampjpe = np.mean(pampjpe)



    return mpjpe,pampjpe

def statistics(df):
    df.drop('time',axis=1, inplace=True, errors='ignore')
    mean_df = df.mean()
    std_df = df.std()
    mean_df.name = "mean"
    std_df.name = "std"
    
    mpjpe = np.sum(df.to_numpy())/(df.shape[0]*df.shape[1]) # Method 1
    return mean_df.mean(), std_df.mean(), mean_df, std_df, mpjpe

def absolute_error_df(ref, src, keypoints):
    absolute_error = pd.DataFrame(src.time) 
    for kp in keypoints:
        p0 = []
        p1 = []
        dist = []
        for i in range(len(ref)):
            p0 = [ref[kp+':X'].loc[i],ref[kp+':Y'].loc[i],ref[kp+':Z'].loc[i]]
            p1 = [src[kp+':X'].loc[i],src[kp+':Y'].loc[i],src[kp+':Z'].loc[i]]
            #assert(euclidean_distance(p0,p1) < 100)
            dist.append(euclidean_distance(p0,p1))
        absolute_error[kp] = dist
        if kp == "test":
            print(dist)
    return absolute_error    

def bone_length(src, keypoints):
    bl = pd.DataFrame(src.time)
    for b in bones.keys():
        if bones[b][0] in keypoints and bones[b][1] in keypoints:
            p0 = []
            p1 = []
            dist = []
            for i in range(len(src)):
                p0 = [src[bones[b][0]+':X'].loc[i],src[bones[b][0]+':Y'].loc[i],src[bones[b][0]+':Z'].loc[i]]
                p1 = [src[bones[b][1]+':X'].loc[i],src[bones[b][1]+':Y'].loc[i],src[bones[b][1]+':Z'].loc[i]]
                dist.append(euclidean_distance(p0,p1))
            bl[b] = dist
    return bl
    
def find_common_keypoints_full(ref,src):
    return [e for e in ref.columns.intersection(src.columns).tolist() if e != 'time' and e != 'JUDGE' and e!= 'tag' and e != 'Unnamed: 0']    

def find_common_keypoints(ref,src):
    return list(set([e.split(':')[0] for e in ref.columns.intersection(src.columns).tolist() if e != 'time' and e != 'Unnamed: 0' ]))


def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def distance_2d_quadratic(x1, x2):
    return abs(x1 - x2)**2

def distance_2d(x1, x2):
    return abs(x1 - x2)

def trim(ref, src):
    tmax = min(ref.time.max(), src.time.max())
    tmin = max(ref.time.min(), src.time.min())
 
    ref = ref[(ref.time >= tmin) & (ref.time <= tmax)]
    src = src[(src.time >= tmin) & (src.time <= tmax)]

    ref.reset_index(drop=True,inplace=True)
    src.reset_index(drop=True,inplace=True)
    
    return ref, src

def interpolation(ref, src):
    ref, src = trim(ref,src)
    new_ref = pd.DataFrame(src.time)
    for column in ref:
        if column != 'time':
            f = interpolate.interp1d(ref.time, ref[column],kind='cubic',fill_value='extrapolate')
            new_ref[column] = f(src.time)
    return new_ref, src

def load_file_pd(file_name):
    return pd.read_csv(file_name, delimiter=',')

def main(args):
    ref_dir = args.ref[0]
    src_dir = args.src[0]
    out_dir = args.out[0]
    list_values = []
    for filename in os.listdir(src_dir):
        print(filename)
        f = os.path.join(src_dir, filename)
        if os.path.isfile(f):
            ref = load_file_pd(os.path.join(ref_dir, filename))
            src = load_file_pd(f)
            ref, src = interpolation(ref,src)
            keypoints = find_common_keypoints(ref,src)
            accel = compute_error_accel(ref,src,keypoints)
            mpjpe,pampjpe = calculate_mpjpe(ref,src,keypoints)
            list_values.append([mpjpe*1000,accel*1000])
    #print(np.mean(np.array(list_values),axis=0))  
    print(np.round(np.mean(np.array(list_values),axis=0),2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="automatic skeletons analyzer", epilog="PARCOLAB")
    parser.add_argument("--ref",
                        "-r",
                        dest="ref",
                        required=True,
                        nargs=1,
                        help="path to the reference keypoints files")
    parser.add_argument("--source",
                        "-s",
                        dest="src",
                        required=True,
                        nargs=1,
                        help="path to the source keypoints files")
    parser.add_argument("--out",
                        "-o",
                        dest="out",
                        required=False,
                        nargs=1,
                        help="path of output files")
    parser.add_argument("--side",
                        "-c",
                        dest="side",
                        required=False,
                        nargs=1,
                        default='full',
                        help="")
    
    args = parser.parse_args()
    main(args)