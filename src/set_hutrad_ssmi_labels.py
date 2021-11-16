#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
set_hutrad_ssmi_labels.py

Create labels for the microwave instrument antennas and orientation.

Created on Tue Oct 26 10:24:18 2021

@author: thayer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Areapoints for each object from point_classifier

hutrad_face = {
    "mosaic_rs_170420.RiSCAN": [
        [
            "ScanPos008",
            15746309
        ],
        [
            "ScanPos008",
            15746234
        ],
        [
            "ScanPos008",
            15746236
        ],
        [
            "ScanPos008",
            15745829
        ],
        [
            "ScanPos008",
            15745916
        ],
        [
            "ScanPos008",
            15735704
        ],
        [
            "ScanPos008",
            15735699
        ],
        [
            "ScanPos008",
            15735690
        ],
        [
            "ScanPos008",
            15735652
        ],
        [
            "ScanPos008",
            15735333
        ],
        [
            "ScanPos008",
            15735294
        ],
        [
            "ScanPos008",
            15735256
        ],
        [
            "ScanPos008",
            15735208
        ],
        [
            "ScanPos008",
            15735188
        ],
        [
            "ScanPos008",
            15735122
        ],
        [
            "ScanPos008",
            15735108
        ],
        [
            "ScanPos008",
            15739834
        ],
        [
            "ScanPos008",
            15739851
        ],
        [
            "ScanPos008",
            15739879
        ],
        [
            "ScanPos008",
            15740074
        ],
        [
            "ScanPos008",
            15740062
        ],
        [
            "ScanPos008",
            15740347
        ],
        [
            "ScanPos008",
            15740570
        ],
        [
            "ScanPos008",
            15740594
        ],
        [
            "ScanPos008",
            15740597
        ],
        [
            "ScanPos008",
            15745015
        ],
        [
            "ScanPos008",
            15745069
        ],
        [
            "ScanPos008",
            15745834
        ],
        [
            "ScanPos008",
            15745138
        ],
        [
            "ScanPos008",
            15745329
        ],
        [
            "ScanPos008",
            15745576
        ],
        [
            "ScanPos008",
            15745591
        ],
        [
            "ScanPos008",
            15746120
        ],
        [
            "ScanPos008",
            15746194
        ]
    ]
}

hutrad_right = {
    "mosaic_rs_170420.RiSCAN": [
        [
            "ScanPos008",
            15745075
        ],
        [
            "ScanPos008",
            15745132
        ],
        [
            "ScanPos008",
            15745127
        ],
        [
            "ScanPos008",
            15745149
        ],
        [
            "ScanPos008",
            15745150
        ],
        [
            "ScanPos008",
            15745160
        ],
        [
            "ScanPos008",
            15745311
        ],
        [
            "ScanPos008",
            15745560
        ],
        [
            "ScanPos008",
            15745370
        ],
        [
            "ScanPos008",
            15745378
        ],
        [
            "ScanPos008",
            15745382
        ],
        [
            "ScanPos008",
            15745395
        ],
        [
            "ScanPos008",
            15745400
        ],
        [
            "ScanPos008",
            15746097
        ],
        [
            "ScanPos008",
            15746155
        ],
        [
            "ScanPos008",
            15746150
        ],
        [
            "ScanPos008",
            15746152
        ]
    ]
}

hutrad_middle = {
    "mosaic_rs_170420.RiSCAN": [
        [
            "ScanPos008",
            15740325
        ],
        [
            "ScanPos008",
            15740320
        ],
        [
            "ScanPos008",
            15740321
        ],
        [
            "ScanPos008",
            15740341
        ],
        [
            "ScanPos008",
            15740433
        ],
        [
            "ScanPos008",
            15740440
        ],
        [
            "ScanPos008",
            15740546
        ],
        [
            "ScanPos008",
            15740555
        ],
        [
            "ScanPos008",
            15740592
        ],
        [
            "ScanPos008",
            15740600
        ],
        [
            "ScanPos008",
            15744986
        ],
        [
            "ScanPos008",
            15744992
        ],
        [
            "ScanPos008",
            15745004
        ],
        [
            "ScanPos008",
            15745058
        ]
    ]
}

hutrad_left = {
    "mosaic_rs_170420.RiSCAN": [
        [
            "ScanPos008",
            15738384
        ],
        [
            "ScanPos008",
            15738386
        ],
        [
            "ScanPos008",
            15738396
        ],
        [
            "ScanPos008",
            15739785
        ],
        [
            "ScanPos008",
            15739795
        ],
        [
            "ScanPos008",
            15739793
        ],
        [
            "ScanPos008",
            15739829
        ],
        [
            "ScanPos008",
            15739960
        ],
        [
            "ScanPos008",
            15740011
        ],
        [
            "ScanPos008",
            15740031
        ],
        [
            "ScanPos008",
            15740052
        ],
        [
            "ScanPos008",
            15740050
        ],
        [
            "ScanPos008",
            15740310
        ]
    ]
}

ssmi_face = {
    "mosaic_rs_170420.RiSCAN": [
        [
            "ScanPos008",
            10636074
        ],
        [
            "ScanPos008",
            10635984
        ],
        [
            "ScanPos008",
            10635775
        ],
        [
            "ScanPos008",
            10635544
        ],
        [
            "ScanPos008",
            10635640
        ],
        [
            "ScanPos008",
            10635617
        ],
        [
            "ScanPos008",
            10635497
        ],
        [
            "ScanPos008",
            10638078
        ],
        [
            "ScanPos008",
            10636800
        ],
        [
            "ScanPos008",
            10636813
        ],
        [
            "ScanPos008",
            10636893
        ],
        [
            "ScanPos008",
            10636878
        ],
        [
            "ScanPos008",
            10637059
        ],
        [
            "ScanPos008",
            10637084
        ],
        [
            "ScanPos008",
            10637094
        ],
        [
            "ScanPos008",
            10637491
        ],
        [
            "ScanPos008",
            10637519
        ],
        [
            "ScanPos008",
            10637365
        ],
        [
            "ScanPos008",
            10637451
        ],
        [
            "ScanPos008",
            10639192
        ],
        [
            "ScanPos008",
            10639191
        ],
        [
            "ScanPos008",
            10639304
        ],
        [
            "ScanPos008",
            10636005
        ]
    ]
}

ssmi_19 = {
    "mosaic_rs_170420.RiSCAN": [
        [
            "ScanPos008",
            10644958
        ],
        [
            "ScanPos008",
            10642772
        ],
        [
            "ScanPos008",
            10642589
        ],
        [
            "ScanPos008",
            10642293
        ],
        [
            "ScanPos008",
            10642322
        ],
        [
            "ScanPos008",
            10643876
        ],
        [
            "ScanPos008",
            10644150
        ],
        [
            "ScanPos008",
            10644416
        ],
        [
            "ScanPos008",
            10645126
        ],
        [
            "ScanPos008",
            10645223
        ],
        [
            "ScanPos008",
            10645393
        ],
        [
            "ScanPos008",
            10645621
        ],
        [
            "ScanPos008",
            10645035
        ]
    ]
}

# Now load the scan these came from
project_path = "../data/RS"
project_name = "mosaic_rs_170420.RiSCAN"

project = pydar.Project(project_path, project_name, 
                      import_mode='read_scan', las_fieldnames=['Points',
                        'PointId', 'Classification'], class_list='all')

project.read_transforms()
project.apply_transforms(['current_transform'])

# Get the coordinates for each set of points
cc_hutrad_face = project.areapoints_to_cornercoords(hutrad_face)
cc_hutrad_right = project.areapoints_to_cornercoords(hutrad_right)
cc_hutrad_middle = project.areapoints_to_cornercoords(hutrad_middle)
cc_hutrad_left = project.areapoints_to_cornercoords(hutrad_left)
cc_ssmi = project.areapoints_to_cornercoords(ssmi_face)
cc_ssmi_19 = project.areapoints_to_cornercoords(ssmi_19)

# %% Find the plane that best fits the hutrad's face

G = np.ones(cc_hutrad_face.shape)
G[:,:2] = cc_hutrad_face[:,:2]
(a, b, c), resid, rank, s = np.linalg.lstsq(G, cc_hutrad_face[:,2])

normal = (a, b, -1)
normal = normal/np.linalg.norm(normal)
point = np.array([0, 0, c])

# %% Project the antenna openings onto plane

# this is equivalent to rotating such that normal aligns with [0, 0, 1]
# this rotation can be broken down into first a rotation around azimuth
# (to bring normal into x-z plane) and then a rotation around inclination
theta = np.arctan2(normal[1], normal[0]) # azimuth
phi = np.arcsin(normal[2]) # inclination
beta = phi - np.pi/2

R_z = np.array([[np.cos(-theta), -np.sin(-theta), 0],
                [np.sin(-theta), np.cos(-theta), 0],
                [0, 0, 1]])
R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])

face_r = (R_y @ R_z @ cc_hutrad_face.T).T
right_r = (R_y @ R_z @ cc_hutrad_right.T).T
middle_r = (R_y @ R_z @ cc_hutrad_middle.T).T
left_r = (R_y @ R_z @ cc_hutrad_left.T).T


# %% Now compute center estimates for each of the openings

def estimate_center(pts):
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) 
        """
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the 
        mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    x = pts[:,0]
    y = pts[:,1]
    
    center_2, ier = optimize.leastsq(f_2, (np.mean(x), np.mean(y)))

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(xc_2, yc_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)
    residu2_2  = sum((Ri_2**2-R_2**2)**2)
    
    return xc_2, yc_2, R_2, residu_2

left = estimate_center(left_r)
middle = estimate_center(middle_r)
right = estimate_center(right_r)

# %% Plot just to check that we've done this correctly
f, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.scatter(x=face_r[:,0], y=face_r[:,1], c=face_r[:,2], s=5, marker='o',
               vmin=-9.58, vmax=-9.52)
ax.scatter(x=right_r[:,0], y=right_r[:,1], c=right_r[:,2], s=9, marker='+',
           vmin=-9.58, vmax=-9.52)
ax.scatter(x=middle_r[:,0], y=middle_r[:,1], c=middle_r[:,2], s=9, marker='*',
           vmin=-9.58, vmax=-9.52)
h = ax.scatter(x=left_r[:,0], y=left_r[:,1], c=left_r[:,2], s=9, marker='s',
           vmin=-9.58, vmax=-9.52)

ax.plot(left[0], left[1], 'o', markersize=10)
ax.plot(middle[0], middle[1], 'o', markersize=10)
ax.plot(right[0], right[1], 'o', markersize=10)

ax.axis('equal')
f.colorbar(h)

# %% reverse the transformation and set the labels

R_tot = R.from_matrix(R_y @ R_z)

left_ctr = R_tot.apply(np.array([left[0], left[1], np.mean(face_r[:,2])]), 
                       inverse=True)
left_ori = R_tot.apply(np.array([left[0], left[1], np.mean(face_r[:,2])-.1]), 
                       inverse=True)
middle_ctr = R_tot.apply(np.array([middle[0], middle[1], np.mean(face_r[:,2])]), 
                       inverse=True)
middle_ori = R_tot.apply(np.array([middle[0], middle[1], np.mean(face_r[:,2])-.1]), 
                       inverse=True)
right_ctr = R_tot.apply(np.array([right[0], right[1], np.mean(face_r[:,2])]), 
                       inverse=True)
right_ori = R_tot.apply(np.array([right[0], right[1], np.mean(face_r[:,2])-.1]), 
                       inverse=True)

ss = project.scan_dict['ScanPos008']
ss.load_labels()
ss.add_label('RS', 'HUTRAD', 'left_ctr', left_ctr[0], left_ctr[1], left_ctr[2])
ss.add_label('RS', 'HUTRAD', 'left_ori', left_ori[0], left_ori[1], left_ori[2])
ss.add_label('RS', 'HUTRAD', 'middle_ctr', middle_ctr[0], middle_ctr[1], middle_ctr[2])
ss.add_label('RS', 'HUTRAD', 'middle_ori', middle_ori[0], middle_ori[1], middle_ori[2])
ss.add_label('RS', 'HUTRAD', 'right_ctr', right_ctr[0], right_ctr[1], right_ctr[2])
ss.add_label('RS', 'HUTRAD', 'right_ori', right_ori[0], right_ori[1], right_ori[2])

# %% Find the plane that best fits the ssmi's face

G = np.ones(cc_ssmi.shape)
G[:,:2] = cc_ssmi[:,:2]
(a, b, c), resid, rank, s = np.linalg.lstsq(G, cc_ssmi[:,2])

normal = (a, b, -1)
normal = normal/np.linalg.norm(normal)
point = np.array([0, 0, c])

# %% Project the antenna openings onto plane

# this is equivalent to rotating such that normal aligns with [0, 0, 1]
# this rotation can be broken down into first a rotation around azimuth
# (to bring normal into x-z plane) and then a rotation around inclination
theta = np.arctan2(normal[1], normal[0]) # azimuth
phi = np.arcsin(normal[2]) # inclination
beta = phi - np.pi/2

R_z = np.array([[np.cos(-theta), -np.sin(-theta), 0],
                [np.sin(-theta), np.cos(-theta), 0],
                [0, 0, 1]])
R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])

ssmi_r = (R_y @ R_z @ cc_ssmi.T).T


# %% Now compute center estimate for the rectangle

def rect_fun(r_param):
    """
    must define x and y as coordinates of points in higher scope

    Parameters
    ----------
    r_param : 1d array
        five element array of [x_center, y_center, x_width, y_height, theta]

    Returns
    -------
    Sum squared distances between points and rectangle

    """
    
    # Transform x and y into rectangle coordinates
    xy = np.vstack([x, y])
    xy -= np.array([[r_param[0]], [r_param[1]]])
    xy = np.array([[np.cos(-r_param[4]), -np.sin(-r_param[4])],
                  [np.sin(-r_param[4]), np.cos(-r_param[4])]]) @ xy
    
    # Get the squared distance from each point to the axis aligned
    # rectangle
    d2 = np.zeros(xy.shape[1])
    for i in np.arange(d2.size):
        x_p = np.abs(xy[0,i])
        y_p = np.abs(xy[1,i])
        if (x_p >= r_param[2]/2) and ((y_p >= r_param[3]/2)):
            # outside rectangle corner, squared dist to corner
            d2[i] = (x_p - r_param[2]/2)**2 + (y_p - r_param[3]/2)**2
        elif (x_p < r_param[2]/2) and ((y_p < r_param[3]/2)):
            # inside rectangle, min dist
            d2[i] = min(r_param[2]/2 - x_p, r_param[3]/2 - y_p)**2
        elif (x_p >= r_param[2]/2):
            # we're to the side of the rectangle
            d2[i] = (x_p - r_param[2]/2)**2
        else:
            # we're above rectangle
            d2[i] = (y_p - r_param[3]/2)**2
    
    return d2

x = ssmi_r[:,0]
y = ssmi_r[:,1]

r_param0 = (x.mean(), y.mean(), 0.6, 0.2, 0)

r_param_opt, ier = optimize.leastsq(rect_fun, r_param0)

# %% Plot just to check that we've done this correctly
f, ax = plt.subplots(1, 1, figsize=(8, 8))

h = ax.scatter(x=ssmi_r[:,0], y=ssmi_r[:,1], c=ssmi_r[:,2], s=5, marker='o')

ax.plot(r_param_opt[0], r_param_opt[1], 'o', markersize=10)

ax.axis('equal')
f.colorbar(h)

# %% reverse the transformation and set the labels

R_tot = R.from_matrix(R_y @ R_z)

ssmi_ctr = R_tot.apply(np.array([r_param_opt[0], r_param_opt[1], 
                                 np.mean(ssmi_r[:,2])]), 
                       inverse=True)
ssmi_ori = R_tot.apply(np.array([r_param_opt[0], r_param_opt[1], 
                                 np.mean(ssmi_r[:,2])-.1]), 
                       inverse=True)

ss = project.scan_dict['ScanPos008']
ss.load_labels()
ss.add_label('RS', 'SSMI', 'ssmi_ctr', ssmi_ctr[0], ssmi_ctr[1], ssmi_ctr[2])
ss.add_label('RS', 'SSMI', 'ssmi_ori', ssmi_ori[0], ssmi_ori[1], ssmi_ori[2])

# %% Repeat for 19 GHz SSMI

# this is equivalent to rotating such that normal aligns with [0, 0, 1]
# this rotation can be broken down into first a rotation around azimuth
# (to bring normal into x-z plane) and then a rotation around inclination
theta = np.arctan2(normal[1], normal[0]) # azimuth
phi = np.arcsin(normal[2]) # inclination
beta = phi - np.pi/2

R_z = np.array([[np.cos(-theta), -np.sin(-theta), 0],
                [np.sin(-theta), np.cos(-theta), 0],
                [0, 0, 1]])
R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])

ssmi_19_r = (R_y @ R_z @ cc_ssmi_19.T).T


# %% Now compute center estimate for the rectangle

x = ssmi_19_r[:,0]
y = ssmi_19_r[:,1]

r_param0 = (x.mean(), y.mean(), 0.4, 0.4, 0)

r_param_opt, ier = optimize.leastsq(rect_fun, r_param0)

# %% Plot just to check that we've done this correctly
f, ax = plt.subplots(1, 1, figsize=(8, 8))

h = ax.scatter(x=ssmi_19_r[:,0], y=ssmi_19_r[:,1], c=ssmi_19_r[:,2], s=5, marker='o')

ax.plot(r_param_opt[0], r_param_opt[1], 'o', markersize=10)

ax.axis('equal')
f.colorbar(h)

# %% reverse the transformation and set the labels

R_tot = R.from_matrix(R_y @ R_z)

ssmi19_ctr = R_tot.apply(np.array([r_param_opt[0], r_param_opt[1], 
                                 np.mean(ssmi_19_r[:,2])]), 
                       inverse=True)
ssmi19_ori = R_tot.apply(np.array([r_param_opt[0], r_param_opt[1], 
                                 np.mean(ssmi_19_r[:,2])-.1]), 
                       inverse=True)

ss = project.scan_dict['ScanPos008']
ss.load_labels()
ss.add_label('RS', 'SSMI', 'ssmi19_ctr', ssmi19_ctr[0], ssmi19_ctr[1],
             ssmi19_ctr[2])
ss.add_label('RS', 'SSMI', 'ssmi19_ori', ssmi19_ori[0], ssmi19_ori[1], 
             ssmi19_ori[2])