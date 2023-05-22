# mocap-refinement

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

This repository includes implementation codes or links to the authors’ original codes of filtering methods for denoising and completing data generated by software platforms for human motion analysis, allowing readers to easily reproduce all the algorithms in different experimental settings.

## Methods
### Classical Methods
- ✅ Simple Moving Average (SMA)
- ✅ Weighted Moving Average (WMA)
- ✅ Exponential Moving Average (EMA)
- ✅ Holt Double Exponential Smoothing Filter (HDE)
- ✅ Butterworth (BF)
- ✅ Least Square Gaussian (LSG)
- ✅ Savitzky–Golay (SG)
- 🔄 Interpolation (INT)

### State Observers
- ✅ Kalman Filter:
    - ✅ with random walk motion model (KF0)
    - ✅ 1th-order (KF1)
    - ✅ 2th-order (KF2)
- 🔄 Extended Kalman Filter (EKF)
- 🔄 Unscented Kalman Filter (UKF)
- 🔄 Tobit Kalman Filter (TKF)
- 🔗 [Bolero-Dynammo](https://github.com/lileicc/dynammo)

### Dimensionality Reduction
- ✅ Truncated Singular Value Decomposition (TSVD)
- ✅ Principal Component Analysis (PCA)
- 🔄 Low-Rank Matrix Completion (LRMC)
- 🔄 Noisy Low-Rank Matrix Completion (NLRMC)
- 🔄 Robust Principal Component Analysis (RPCA)
- 🔄 Non-negative Matrix Factorization (NMF)
- 🔄 Dictionary Learning (DL)

### Neural Networks
- 🔗 Autoencoder ([Smoothnet](https://github.com/cure-lab/SmoothNet) - SN)

### Hybrid Approaches
- ✅ Kalman Filter + Differential Evolutionary ([Das2017](https://ieeexplore.ieee.org/document/7996969) - KF+DE)

## Accuracy Results on Human3.6M

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2">denoising</th>
    <th colspan="2">completion</th>
    <th colspan="2">recovery</th>
  </tr>
  <tr>
    <th>MPJPE<br>(mm)</th>
    <th>Accel<br>(mm/s<sup>2</sup>)</th>
    <th>MPJPE<br>(mm)</th>
    <th>Accel<br>(mm/s<sup>2</sup>)</th>
    <th>MPJPE<br>(mm)</th>
    <th>Accel<br>(mm/s<sup>2</sup>)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>baseline</td>
    <td>159.59</td>
    <td>390.78</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>SMA</td>
    <td>58.73</td>
    <td>16.74</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>WMA</td>
    <td>44.77</td>
    <td>12.68</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>EMA</td>
    <td>92.83</td>
    <td>145.66</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>HDE</td>
    <td>105.88</td>
    <td>156.72</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Butterworth</td>
    <td>56.78</td>
    <td>15.46</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Least-Squared</td>
    <td>55.44</td>
    <td>16.17</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Savitzky-Golay</td>
    <td>74.08</td>
    <td>49.99</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Interpolation</td>
    <td>-</td>
    <td>-</td>
    <td>0.02</td>
    <td>0.09</td>
    <td>162.18</td>
    <td>350.38</td>
  </tr>
  <tr>
    <td>KF0</td>
    <td>64.81</td>
    <td>49.96</td>
    <td>28.97</td>
    <td>2.14</td>
    <td>67.01</td>
    <td>49.45</td>
  </tr>
  <tr>
    <td>KF1</td>
    <td>64.28</td>
    <td>53.73</td>
    <td>24.24</td>
    <td>2.02</td>
    <td>66.42</td>
    <td>53.26</td>
  </tr>
  <tr>
    <td>KF2</td>
    <td>65.8</td>
    <td>56.52</td>
    <td>22.67</td>
    <td>2.03</td>
    <td>68.05</td>
    <td>56.14</td>
  </tr>
  <tr>
    <td>T-SVD</td>
    <td>111.42</td>
    <td>106.33</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
    <tr>
    <td>PCA</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>KF+DE</td>
    <td>131.9</td>
    <td>185.76</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
    <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>
