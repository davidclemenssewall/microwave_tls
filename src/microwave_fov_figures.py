#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
microwave_fov_figures.py

Created on Tue Oct 26 16:11:23 2021

@author: thayer
"""

import pandas as pd
import matplotlib.pyplot as plt
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

# %% Extract z values of points

df['pts_z'] = None

for i in np.arange(df.shape[0]):
    df.at[i, 'pts_z'] = vtk_to_numpy(df.at[i, 'points'].GetPoints()
                                     .GetData())[:,2]
    print(df.at[i, 'pts_z'].shape)

# %% Now create plots looking at how the surface changed

f, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(20, 8))

axs = np.ravel(axs)

df_titles = df[['frequency', 'polarization']].drop_duplicates().reset_index(
    drop=True)

for i in range(df_titles.shape[0]):
    # Apr. 17
    frequency = df_titles.at[i, "frequency"]
    polarization = df_titles.at[i, "polarization"]
    pts = df.query('project_name == "mosaic_rs_170420.RiSCAN" and '
                   'frequency == @frequency and '
                   'polarization == @polarization').pts_z
    m0 = np.mean(pts.values[0])
    axs[i].hist(pts, density=True, color='b', alpha=.7)
    
    # Apr. 22
    frequency = df_titles.at[i, "frequency"]
    polarization = df_titles.at[i, "polarization"]
    pts = df.query('project_name == "mosaic_rs_220420.RiSCAN" and '
                   'frequency == @frequency and '
                   'polarization == @polarization').pts_z
    m1 = np.mean(pts.values[0])
    axs[i].hist(pts, density=True, color='r', alpha=.7)
    axs[i].set_title(frequency + ' pol: ' + polarization)
    axs[i].text(.6, .6, "mean change:\n" + str(round(m1-m0, 2)) + " m", 
                transform=axs[i].transAxes)
    
axs[0].set_ylim([0, 50])

axs[4].set_xlabel('Surface Height (m)')
axs[4].set_ylabel('PDF Density')

f.savefig(os.path.join('..', 'figures', 'per_beam_heights.png'))

# %% Add columns corresponding to statistics on pts_z

df['point count'] = df['pts_z'].apply(lambda x: x.shape[0])
df['mean height'] = df['pts_z'].apply(lambda x: np.mean(x))
df['std height'] = df['pts_z'].apply(lambda x: np.std(x))
df['date'] = df['project_name'].apply(lambda x: pydar.mosaic_date_parser(x))

print(df[['date', 'frequency', 'polarization', 'point count', 'mean height', 
    'std height']])

# %% helper funciton

# Define function for writing the camera position and focal point to
# std out when the user presses 'u'
def cameraCallback(obj, event):
    print("Camera Pos: " + str(obj.GetRenderWindow().
                                   GetRenderers().GetFirstRenderer().
                                   GetActiveCamera().GetPosition()))
    print("Focal Point: " + str(obj.GetRenderWindow().
                                    GetRenderers().GetFirstRenderer().
                                    GetActiveCamera().GetFocalPoint()))
    print("Roll: " + str(obj.GetRenderWindow().
                                    GetRenderers().GetFirstRenderer().
                                    GetActiveCamera().GetRoll()))


# %% Examine beams in 3D rendering, particularly, did right beam change

z_min = -2.35
z_max = -1.85

#beam_h = 2

pdata = scan_area.project_dict['mosaic_rs_220420.RiSCAN'].get_merged_points()

# Create vertices
vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
vertexGlyphFilter.SetInputData(pdata)
vertexGlyphFilter.Update()

# # Create elevation filter
elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputConnection(vertexGlyphFilter.GetOutputPort())
# needed to prevent simpleelevationfilter from overwriting 
# Classification array
elevFilter.Update()


# Create mapper, hardcode LUT for now
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
mapper.SetScalarRange(z_min, z_max)
mapper.SetScalarVisibility(1)

# Create Actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create renderer
renderer = vtk.vtkRenderer()

renderer.AddActor(actor)

# Add beams as a cone sources
for i in [1, 5, 9, 13, 15]:
    coneSource = vtk.vtkConeSource()
    coneSource.SetAngle(df.at[i, 'beam_width']/2)
    ctr = np.array(df.at[i, 'ctr'])
    ori = np.array(df.at[i, 'ori'])
    
    # Set the beam height as the greatest distance from beam to point
    beam_h = np.sqrt(np.mean(((vtk_to_numpy(df.at[i, 'points'].GetPoints().GetData()) 
                       - ctr)**2).sum(axis=1)))
    coneSource.SetHeight(beam_h)
    coneSource.SetCenter(ctr - beam_h*10*(ori - ctr)/2)
    coneSource.SetDirection(ori - ctr)
    coneSource.SetResolution(50)
    coneSource.CappingOff()
    coneSource.Update()
    
    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(coneSource.GetOutputPort())
    
    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)
    coneActor.GetProperty().SetOpacity(0.5)
    renderer.AddActor(coneActor)

scalarBar = vtk.vtkScalarBarActor()
scalarBar.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
renderer.AddActor2D(scalarBar)

# Create RenderWindow and interactor, set style to trackball camera
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(1500, 1000)
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)
    
iren.Initialize()
renderWindow.Render()

iren.AddObserver('UserEvent', cameraCallback)
iren.Start()

# %% repeat to save snapshot

Camera_Pos = (7.633547124715694, 4.645905542839899, -0.09970566876032372)
Focal_Point = (20.93980822692692, 3.3339772969789756, -2.179099777456295)
Roll = 91.47122262160124

z_min = -2.35
z_max = -1.85

#beam_h = 2

pdata = scan_area.project_dict['mosaic_rs_220420.RiSCAN'].get_merged_points()

# Create vertices
vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
vertexGlyphFilter.SetInputData(pdata)
vertexGlyphFilter.Update()

# # Create elevation filter
elevFilter = vtk.vtkSimpleElevationFilter()
elevFilter.SetInputConnection(vertexGlyphFilter.GetOutputPort())
# needed to prevent simpleelevationfilter from overwriting 
# Classification array
elevFilter.Update()


# Create mapper, hardcode LUT for now
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(elevFilter.GetOutputPort())
mapper.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
mapper.SetScalarRange(z_min, z_max)
mapper.SetScalarVisibility(1)

# Create Actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create renderer
renderer = vtk.vtkRenderer()

renderer.AddActor(actor)

# Add beams as a cone sources
for i in [1, 5, 9, 13, 15]:
    coneSource = vtk.vtkConeSource()
    coneSource.SetAngle(df.at[i, 'beam_width']/2)
    ctr = np.array(df.at[i, 'ctr'])
    ori = np.array(df.at[i, 'ori'])
    
    # Set the beam height as the greatest distance from beam to point
    beam_h = np.sqrt(np.mean(((vtk_to_numpy(df.at[i, 'points'].GetPoints().GetData()) 
                       - ctr)**2).sum(axis=1)))
    coneSource.SetHeight(beam_h)
    coneSource.SetCenter(ctr - beam_h*10*(ori - ctr)/2)
    coneSource.SetDirection(ori - ctr)
    coneSource.SetResolution(50)
    coneSource.CappingOff()
    coneSource.Update()
    
    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(coneSource.GetOutputPort())
    
    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)
    coneActor.GetProperty().SetOpacity(0.5)
    renderer.AddActor(coneActor)

scalarBar = vtk.vtkScalarBarActor()
scalarBar.SetLookupTable(pydar.mplcmap_to_vtkLUT(z_min, z_max))
renderer.AddActor2D(scalarBar)

# Create RenderWindow
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(1500, 1000)
renderWindow.AddRenderer(renderer)
# Create Camera
camera = vtk.vtkCamera()
camera.SetFocalPoint(Focal_Point)
camera.SetPosition(Camera_Pos)
camera.SetRoll(Roll)
renderer.SetActiveCamera(camera)

renderWindow.Render()

# Screenshot image to save
w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(renderWindow)
w2if.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName(os.path.join('..', 'figures', 'april_22_w_beams.png'))
writer.SetInputData(w2if.GetOutput())
writer.Write()

renderWindow.Finalize()
del renderWindow

# %% Finally, let's look at incident angles, first we need to create normals

radius = 0.1
max_nn = 10

for project_name in project_names:
    scan_area.project_dict[project_name].create_normals(radius=radius,
                                                        max_nn=max_nn)

# %% Repeat point extraction, now this brings normals

# left and right are from the perspective looking in the direction
# that the instrument is looking!!

labels = ss.get_labels()

df_n = pd.DataFrame({'subcategory': ['HUTRAD',
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

# Cartesian product df_n
df_n = df_n.merge(pd.DataFrame({'project_name': project_names}), how='cross')

# and merge to get dataframe we want
df_n = df_n.merge(labels)

# %% Let's extract points for each beam

df_n['points'] = None

for i in np.arange(df_n.shape[0]):
    # Cone is only oriented along x axis, so we need to create the appropriate 
    # transform to align beam axis with x axis.
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(-np.array(df_n.at[i, 'ctr']))
    vec = np.array(df_n.at[i, 'ori'])-np.array(df_n.at[i, 'ctr'])
    transform.RotateZ(-np.arctan2(vec[1], vec[0])*180/np.pi)
    transform.RotateY(np.arcsin(vec[2]/.1)*180/np.pi)
    
    # Create cone, vtk's cone angle is half the beam width I think
    cone = vtk.vtkCone()
    cone.SetTransform(transform)
    cone.SetAngle(df_n.at[i,'beam_width']/2)
    
    # Extract points inside this cone
    extractPoints = vtk.vtkExtractPoints()
    extractPoints.SetImplicitFunction(cone)
    extractPoints.SetInputData(scan_area.project_dict[df_n.at[i,'project_name']]
                               .get_merged_points())
    extractPoints.Update()
    
    df_n.at[i, 'points'] = extractPoints.GetOutput()

# %% Now compute the incidence angle for each point/beam

df_n['incidence angle'] = None

for i in np.arange(df_n.shape[0]):
    ctr = np.array(df.at[i, 'ctr'])
    
    pts = vtk_to_numpy(df_n.at[i, 'points'].GetPoints().GetData())
    vec = pts - ctr
    vec = vec/np.sqrt((vec**2).sum(axis=1))[:, np.newaxis]
    nrm = vtk_to_numpy(df_n.at[i, 'points'].GetPointData().GetNormals())
    
    df_n.at[i, 'incidence angle'] = (np.arccos((vec*nrm).sum(axis=1))*180
                                     /np.pi) - 90

# %% now plot

f, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(20, 8))

axs = np.ravel(axs)

df_titles = df_n[['frequency', 'polarization']].drop_duplicates().reset_index(
    drop=True)

for i in range(df_titles.shape[0]):
    # Apr. 17
    frequency = df_titles.at[i, "frequency"]
    polarization = df_titles.at[i, "polarization"]
    inc = df_n.query('project_name == "mosaic_rs_170420.RiSCAN" and '
                   'frequency == @frequency and '
                   'polarization == @polarization')['incidence angle']
    m0 = np.mean(inc.values[0])
    axs[i].hist(inc, density=True, color='b', alpha=.7)
    
    # Apr. 22
    frequency = df_titles.at[i, "frequency"]
    polarization = df_titles.at[i, "polarization"]
    inc = df_n.query('project_name == "mosaic_rs_220420.RiSCAN" and '
                   'frequency == @frequency and '
                   'polarization == @polarization')['incidence angle']
    m1 = np.mean(inc.values[0])
    axs[i].hist(inc, density=True, color='r', alpha=.7)
    axs[i].set_title(frequency + ' pol: ' + polarization)
    axs[i].text(.2, .6, "mean change:\n" + str(round(m1-m0, 2)) + " deg", 
                transform=axs[i].transAxes)
    
axs[0].set_ylim([0, 0.4])

axs[4].set_xlabel('Incidence Angle (degrees)')
axs[4].set_ylabel('PDF Density')

f.savefig(os.path.join('..', 'figures', 'per_beam_incidence_angle.png'))

# %% and print table again

df_n['point count'] = df_n['incidence angle'].apply(lambda x: x.shape[0])
df_n['mean inc angle'] = df_n['incidence angle'].apply(lambda x: np.mean(x))
df_n['std inc angle'] = df_n['incidence angle'].apply(lambda x: np.std(x))
df_n['date'] = df_n['project_name'].apply(lambda x: pydar.mosaic_date_parser(x))

print(df_n[['date', 'frequency', 'polarization', 'mean inc angle', 
    'std inc angle']])