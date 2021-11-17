#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
microwave_footprint_inc_angle_maps

Created on Wed Nov 17 12:21:14 2021

@author: thayer
"""

import copy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar


# SSMI inc angle
ssmi_inc = 45

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

# Create normals for looking and incident angles
radius = 0.1
max_nn = 10

for project_name in project_names:
    scan_area.project_dict[project_name].create_normals(radius=radius,
                                                        max_nn=max_nn)

# Get the labels
ss = scan_area.project_dict["mosaic_rs_170420.RiSCAN"].scan_dict[
    'ScanPos008']
ss.load_labels()

labels = ss.get_labels()

# %% Create a dataframe to organize this

df = pd.DataFrame({'subcategory': ['HUTRAD',
                                   'HUTRAD',
                                   'HUTRAD',
                                   'HUTRAD',
                                   'HUTRAD',
                                   'HUTRAD',
                                   'SSMI',
                                   'SSMI'],
                   'location': ['left',
                                'left',
                                'middle',
                                'middle',
                                'right',
                                'right',
                                'ssmi',
                                'ssmi19'],
                   'frequency': ['10.7 GHz',
                                 '10.7 GHz',
                                 '18.7 GHz',
                                 '18.7 GHz',
                                 '6.9 GHz',
                                 '6.9 GHz',
                                 '89 GHz',
                                 '19 GHz'],
                   'polarization': ['V',
                                    'H',
                                    'V',
                                    'H',
                                    'V',
                                    'H',
                                    'None',
                                    'None'],
                   'beam_width': [9.1,
                                  6.6,
                                  8.6,
                                  6.4,
                                  14.8,
                                  11.2,
                                  5.88,
                                  6.0]})

# %% modify labels dataframe accordingly

labels = labels.reset_index()
labels['point'] = np.vstack((labels['x_trans'], labels['y_trans'], 
                             labels['z_trans'])).T.tolist()
labels.drop(columns=['category', 'project_name', 'scan_name', 'x', 'y', 'z',
                     'x_trans', 'y_trans', 'z_trans']
            , inplace=True)

labels['location'] = labels['id'].apply(lambda x: x.split('_')[0])
labels['type'] = labels['id'].apply(lambda x: x.split('_')[1])

labels.drop(columns=['id'], inplace=True)

labels = labels.pivot(index=['subcategory', 'location'], columns='type')
labels.columns = labels.columns.droplevel()

labels = labels.reset_index()

# Janna and Philip asked for SSMI measurements to all be made from incidence
# angle of 55 degrees, simplest way to do this is to adjust the ori points
for i in np.arange(labels.shape[0]):
    if labels.at[i, 'subcategory']=='SSMI':
        #Adjust the z value of the orientation point such that it is 55 degre
        # above ctr point. the distance between points is 0.1 m
        labels.at[i, 'ori'][2] = (0.1*np.sin(ssmi_inc*np.pi/180) + 
                                  labels.at[i,'ctr'][2])

# Cartesian product df
df = df.merge(pd.DataFrame({'project_name': project_names}), how='cross')

# and merge to get dataframe we want
df = df.merge(labels)

# %% Let's extract points for each beam

df['points'] = None

for i in np.arange(df.shape[0]):
    # Cone is only oriented along x axis, so we need to create the appropriate 
    # transform to align beam axis with x axis.
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(-np.array(df.at[i, 'ctr']))
    vec = np.array(df.at[i, 'ori'])-np.array(df.at[i, 'ctr'])
    transform.RotateZ(-np.arctan2(vec[1], vec[0])*180/np.pi)
    transform.RotateY(np.arcsin(vec[2]/.1)*180/np.pi)
    
    # Create cone, vtk's cone angle is half the beam width I think
    cone = vtk.vtkCone()
    cone.SetTransform(transform)
    cone.SetAngle(df.at[i,'beam_width']/2)
    
    # Extract points inside this cone
    extractPoints = vtk.vtkExtractPoints()
    extractPoints.SetImplicitFunction(cone)
    extractPoints.SetInputData(scan_area.project_dict[df.at[i,'project_name']]
                               .get_merged_points())
    extractPoints.Update()
    
    df.at[i, 'points'] = extractPoints.GetOutput()

# %% Now compute the incidence angle for each point/beam

df['incidence angle'] = None

for i in np.arange(df.shape[0]):
    ctr = np.array(df.at[i, 'ctr'])
    
    pts = vtk_to_numpy(df.at[i, 'points'].GetPoints().GetData())
    vec = pts - ctr
    vec = vec/np.sqrt((vec**2).sum(axis=1))[:, np.newaxis]
    nrm = vtk_to_numpy(df.at[i, 'points'].GetPointData().GetNormals())
    
    df.at[i, 'incidence angle'] = (np.arccos((vec*nrm).sum(axis=1))*180
                                     /np.pi) - 90

# %% Useful functions for binning and plotting

# Note, the cython version of this function is much faster, fortunately these
# pointclouds are small...
def create_counts_means_M2_cy(nbin_0, nbin_1, Points, xy):
    """
    Return the binwise number of points, mean, and M2 estimate of the sum of
    squared deviations (using Welford's algorithm)
                        
    from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    
    Parameters
    ----------
    nbin_0 : long
        Number of bins along the zeroth axis
    nbin_1 : long
        Number of bins along the first axis
    Points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    xy : long[:]
        Bin index for each point, must be same as numbe rof points.

    Returns:
    --------
    counts : float[:]
        Array with the counts for each bin. Length nbin_0*nbin_1
    means : float[:]
        Array with the mean of z values for each bin. Length nbin_0*nbin_1
    m2s : float[:]
        Array with the M2 estimate of the sum of squared deviations
        Length nbin_0*nbin_1
    """

    # Chose to make this float for division purposes but could have unexpected
    # results, check if so
    counts = np.zeros(nbin_0 * nbin_1, dtype=np.float32)
    
    means = np.zeros(nbin_0 * nbin_1, dtype=np.float32)

    m2s = np.zeros(nbin_0 * nbin_1, dtype=np.float32)

    for i in range(len(xy)):
        counts[xy[i]] += 1
        delta = Points[i, 2] - means[xy[i]]
        means[xy[i]] += delta / counts[xy[i]]
        delta2 = Points[i, 2] - means[xy[i]]
        m2s += delta*delta2
    
    return counts, means, m2s

def gridded_counts_means_vars(points, edges):
    """
    Grids a point could in x and y and returns the cellwise counts, means and
    variances.

    Parameters
    ----------
    points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    edges : list
        2 item list containing edges for gridding

    Returns:
    --------
    counts : long[:, :]
        Gridded array with the counts for each bin
    means : float[:, :]
        Gridded array with the mean z value for each bin.
    vars : float[:, :]
        Gridded array with the variance in z values for each bin.

    """

    Ncount = tuple(np.searchsorted(edges[i], points[:,i], 
                               side='right') for i in range(2))

    nbin = np.empty(2, np.int_)
    nbin[0] = len(edges[0]) + 1
    nbin[1] = len(edges[1]) + 1

    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute gridded mins and counts
    counts, means, m2s = create_counts_means_M2_cy(nbin[0], nbin[1],
                                                     points,
                                                     np.int_(xy))
    counts = counts.reshape(nbin)
    means = means.reshape(nbin)
    m2s = m2s.reshape(nbin)
    core = 2*(slice(1, -1),)
    counts = counts[core]
    means = means[core]
    m2s = m2s[core]
    means[counts==0] = np.nan
    m2s[counts==0] = np.nan
    var = m2s/counts

    return counts, means, var

# %% for each set of points, create polar plot bins

df['pol_cts'] = None
df['pol_mean'] = None
df['pol_var'] = None

# angular bin widths
r_edges = np.linspace(0, 8*np.pi/180, num=9)
t_edges = np.linspace(-np.pi, np.pi, num=19)

#i = 13

for i in np.arange(df.shape[0]):
    elevFilter = vtk.vtkSimpleElevationFilter()
    elevFilter.SetInputData(df.at[i, 'points'])
    elevFilter.Update()
    
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(-np.array(df.at[i, 'ctr']))
    vec = np.array(df.at[i, 'ori'])-np.array(df.at[i, 'ctr'])
    transform.RotateZ(-np.arctan2(vec[1], vec[0])*180/np.pi)
    transform.RotateY(np.arcsin(vec[2]/.1)*180/np.pi)
    
    transformFilter = vtk.vtkTransformFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(elevFilter.GetOutput())
    transformFilter.Update()
    
    spTrans = vtk.vtkSphericalTransform().GetInverse()
    tf2 = vtk.vtkTransformFilter()
    tf2.SetTransform(spTrans)
    tf2.SetInputData(transformFilter.GetOutput())
    tf2.Update()
    
    pts = vtk_to_numpy(tf2.GetOutput().GetPoints().GetData())
    inc_angles = df.at[i, 'incidence angle']
    
    pts[:,1] -= np.pi/2
    pts[:,2] -= np.pi
    pts[:,2] *= -1
    
    #plt.scatter(pts[:,1]-np.pi/2, pts[:,2]-np.pi, c=elev)
    #plt.colorbar()
    #plt.axis('equal')
    
    pol_pts = np.vstack((np.sqrt(np.square(pts[:,1:]).sum(axis=1)),
                             np.arctan2(pts[:,2], pts[:,1]),
                             inc_angles)).T
    
    df.at[i,'pol_cts'], df.at[i,'pol_mean'], df.at[i,'pol_var'] = (
        gridded_counts_means_vars(pol_pts, (r_edges, t_edges)))


# %% 

cmap_div = copy.copy(cm.get_cmap('RdBu_r'))
cmap_div.set_bad(color='black')
cmap_seq = copy.copy(cm.get_cmap('rainbow'))
cmap_seq.set_bad(color='black')

warnings.warn('Hackish way to cycle through dataframe, check if modifying')
for i in np.arange(8)*2:
    f, axs = plt.subplots(1, 3, figsize=(15,5), 
                          subplot_kw=dict(projection='polar'))
    
    apr17 = df.at[i,'pol_mean']
    apr22 = df.at[i+1, 'pol_mean']
    hmin = min(np.nanmin(apr17), np.nanmin(apr22))
    hmax = max(np.nanmax(apr17), np.nanmax(apr22))
    
    h = axs[0].pcolormesh(t_edges, r_edges*180/np.pi, 
              apr17,
              cmap=cmap_seq, vmin=hmin, vmax=hmax)

    axs[0].set_ylim([0, df.at[i, 'beam_width']/2])
    axs[0].set_title(pydar.mosaic_date_parser(df.at[i,'project_name']))
    f.colorbar(h, ax=axs[0], shrink=0.8, label='Inc. Angle ($^o$)',
               format='%.3f')
    axs[0].yaxis.set_ticks(r_edges[r_edges<=df.at[i, 'beam_width']*np.pi/180/2]*180/np.pi)
    axs[0].tick_params(axis='y', colors='lime')
    
    
    h = axs[1].pcolormesh(t_edges, r_edges*180/np.pi, 
              apr22,
              cmap=cmap_seq, vmin=hmin, vmax=hmax)

    axs[1].set_ylim([0, df.at[i+1, 'beam_width']/2])
    axs[1].set_title(pydar.mosaic_date_parser(df.at[i+1,'project_name']))
    f.colorbar(h, ax=axs[1], shrink=0.8, label='Inc. Angle ($^o$)',
               format='%.3f')
    axs[1].yaxis.set_ticks(r_edges[r_edges<=df.at[i, 'beam_width']*np.pi/180/2]*180/np.pi)
    axs[1].tick_params(axis='y', colors='lime')
    
    diff_max = np.nanmax(np.abs(apr22-apr17))
    h = axs[2].pcolormesh(t_edges, r_edges*180/np.pi, 
              apr22-apr17,
              cmap=cmap_div, vmin=-diff_max, vmax=diff_max)

    axs[2].set_ylim([0, df.at[i+1, 'beam_width']/2])
    axs[2].set_title('Difference')
    f.colorbar(h, ax=axs[2], shrink=0.8, label='$\Delta$ Inc. Angle ($^o$)',
               format='%.3f')
    axs[2].yaxis.set_ticks(r_edges[r_edges<=df.at[i, 'beam_width']*np.pi/180/2]*180/np.pi)
    axs[2].tick_params(axis='y', colors='lime')
    
    f.suptitle(df.at[i, 'frequency'] + ' pol: ' + df.at[i, 'polarization'])
    
    f.savefig(os.path.join('..', 'figures', 'polar_map_inc_angle_' 
                           + df.at[i, 'frequency'].split(' ')[0] + '_pol_' +
                           df.at[i,'polarization'] + '.png'))

# %% write out data

np.save(os.path.join('..', 'data', 'polar_gridded_data','r_edges.npy'),
        r_edges)
np.save(os.path.join('..', 'data', 'polar_gridded_data','t_edges.npy'),
        t_edges)

df[['subcategory', 'location', 'frequency', 'polarization', 'beam_width',
    'project_name', 'ctr', 'ori', 'pol_cts', 'pol_mean', 'pol_var']
   ].to_pickle(os.path.join('..', 'data', 'polar_gridded_data',
                                       'polar_inc_angle_maps.pkl'))