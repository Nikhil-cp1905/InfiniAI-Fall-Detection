import numpy as np

def extract_pose_features(kpts):
    # kpts: (17, 3)
    kp = kpts[:, :2]

    head = kp[0]
    lhip = kp[11]
    rhip = kp[12]
    hip = (lhip + rhip) / 2

    vertical_dist = abs(head[1] - hip[1])

    dx = head[0] - hip[0]
    dy = head[1] - hip[1]
    angle = np.arctan2(dy, dx)

    return np.array([vertical_dist, angle])

