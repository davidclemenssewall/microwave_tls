#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
microwave_glazing_figure.py

Create a couple figures showing the reflectance Apr. 17 and Apr. 22

Created on Tue Nov 16 09:28:23 2021

@author: thayer
"""

import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Load scans
project_path = "../data/RS"
project_names = ["mosaic_rs_170420.RiSCAN",
                 "mosaic_rs_220420.RiSCAN",]

scan_area = pydar.ScanArea(project_path, project_names=project_names, 
                           import_mode='read_scan',
                        las_fieldnames=['Points', 'PointId', 'Classification',
                                        'Reflectance'], class_list='all')

for project_name in project_names:
    scan_area.project_dict[project_name].read_transforms()
    scan_area.project_dict[project_name].apply_transforms([
        'current_transform'])
    
# %% Examine scan in map view

vmin = -20
vmax = -10

scan_area.project_dict[project_names[1]].display_project(vmin, vmax,
                                                         field='Reflectance',
                                                         mapview=True)

# %% Save images

camera_pos = (15, 4.520507966818727, 500.0)
foc_point = (15, 4.520507966818727, -5.0)
roll = 22.974745361393772

image_scale = 10
window_size = (700, 700)

scan_area.project_dict[project_names[0]].project_to_image(vmin, vmax, 
    foc_point, camera_pos, roll=roll, image_scale=image_scale, mode='map',
    colorbar=True, field='Reflectance', window_size=window_size,
    path=os.path.join('..', 'figures', 'apr17_reflectance.png'))

scan_area.project_dict[project_names[1]].project_to_image(vmin, vmax, 
    foc_point, camera_pos, roll=roll, image_scale=image_scale, mode='map',
    colorbar=True, field='Reflectance', window_size=window_size,
    path=os.path.join('..', 'figures', 'apr22_reflectance.png'))