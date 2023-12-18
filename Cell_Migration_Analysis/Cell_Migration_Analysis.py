#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.interactive(False)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (
                            FigureCanvasQTAgg as FigureCanvas,
                            NavigationToolbar2QT as NavigationToolbar
                            )
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from pathlib import Path
from multiprocessing import Process
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QIntValidator, QMouseEvent
from PyQt5.QtWidgets import (
                            QApplication, QLabel, QWidget,
                            QPushButton, QHBoxLayout, QVBoxLayout,
                            QComboBox, QCheckBox, QSlider, QProgressBar,
                            QFormLayout, QLineEdit, QTabWidget,
                            QSizePolicy, QFileDialog, QMessageBox
                            )

################################################################################
# Arrow3D definition extending FancyArrowPatch for 3D arrowheads.
################################################################################

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

################################################################################

def process_file ( datafile ):
    if datafile.suffix == '.csv':
        position_data = parse_datafile(datafile)
        # position_data = correct_drift(position_data)
        metrics = calculate_metrics(position_data)
        # time_length = position_data.x.shape[1]
        data_format = np.dtype([ ('i', int),
                                  ('x', object),
                                  ('d', float, 3),
                                  ('p', float, 3),
                                  ('m', float, 3),
                                  ('avg', float, 3),
                                  ('corr', float),
                                  ('curl', float, 3) ])
        data = np.empty(np.unique(position_data.i).size,
                        dtype=data_format).view(np.recarray)
        data.i = position_data.i
        data.x = position_data.x
        data.d = metrics.d
        data.p = metrics.p
        data.m = metrics.m
        # delta_x = np.empty(np.unique(position_data.i).size, float, 3)
        delta_x = np.empty((np.unique(position_data.i).size, 3))
        end_x = np.empty((np.unique(position_data.i).size, 3))
        for index in np.unique(position_data.i):
            delta_x[index] = data[index].x[-1] - data[index].x[0]
            end_x[index]=data[index].x[-1]
        # print(delta_x)
        data.corr, data.avg = calculate_correlation(end_x, delta_x)
        data.curl = calculate_curl(end_x, delta_x)
        
        with open(datafile.with_suffix('.pkl'),'wb') as outstream:
            pickle.dump(data, outstream)
        return data
    elif datafile.suffix == '.pkl':
        with open(datafile,'rb') as instream:
            data = pickle.load(instream)
        return data
    else:
        print('Unknown data file format. Exiting.')
        return None

################################################################################

def parse_datafile ( datafile ):
    file_data_format = np.dtype([ ('i', int), ('t', int),
                              ('x', float, 3) ])
    data = np.genfromtxt(datafile,
                            delimiter = ',', # names = True,
                            comments = '#',
                            skip_header = 1,
                            usecols = (4,3,0,1,2),
                            dtype = file_data_format ).view(np.recarray)
    data.i = data.i - np.amin(data.i)
    data.t = data.t - np.amin(data.t)
    time_min = 20
    last_time = np.amax(data.t)
    data = np.sort(data, order = ['i', 't'])
    # Only want ones that were tracked for the entire time.
    # for time in range(time_min+1,last_time+1):
    #     data = np.delete(data, np.argwhere(data.t == time))
    index = 0
    while index <= np.amax(data.i):
        if data[data.i == index].size < time_min+1:
            data = np.delete(data, np.argwhere(data.i == index))
            data.i = np.where(data.i > index, data.i-1, data.i)
        else:
            index += 1
    data_format = np.dtype([ ('i', int), ('x', object) ])
    restructured_data = np.zeros(np.unique(data.i).size,
                                        dtype  = data_format).view(np.recarray)
    for index in np.unique(data.i):
        restructured_data[index] = (index, data[data.i==index].x)
    # restructured_data = []
    # for index in np.unique(data.i):
    #     subset_data = data[data.i == index]
    #     entry = {'i': index, 'x': subset_data.x}
    #     restructured_data.append(entry)
    # print ([entry['i'] for entry in restructured_data])
    # print (restructured_data.x)
    return restructured_data

################################################################################
# to do time range reset
def re_parse_datafile ( datafile , time_min , time_max):
    file_data_format = np.dtype([ ('i', int), ('t', int),
                              ('x', float, 3) ])
    data = np.genfromtxt(datafile,
                            delimiter = ',', # names = True,
                            comments = '#',
                            skip_header = 1,
                            usecols = (4,3,0,1,2),
                            dtype = file_data_format ).view(np.recarray)
    data.i = data.i - np.amin(data.i)
    data.t = data.t - np.amin(data.t)
    time_min = 30
    data = np.sort(data, order = ['i', 't'])
    # Only want ones that were tracked for the entire time.
    for time in range(time_min+1,time_max+1):
        data = np.delete(data, np.argwhere(data.t == time))
    # index = 0
    # while index <= np.amax(data.i):
    #     if data[data.i == index].size < time_min+1:
    #         data = np.delete(data, np.argwhere(data.i == index))
    #         data.i = np.where(data.i > index, data.i-1, data.i)
    #     else:
    #         index += 1
    data_format = np.dtype([ ('i', int), ('x', object) ])
    restructured_data = np.zeros(np.unique(data.i).size,
                                        dtype  = data_format).view(np.recarray)
    for index in np.unique(data.i):
        restructured_data[index] = (index, data[data.i==index].x)
    return restructured_data

################################################################################

def re_process_file ( datafile , time_min , time_max ):
    position_data = re_parse_datafile(datafile , time_min , time_max)
    # position_data = correct_drift(position_data)
    metrics = calculate_metrics(position_data)
    # time_length = position_data.x.shape[1]
    data_format = np.dtype([ ('i', int),
                                  ('x', object),
                                  ('d', float, 3),
                                  ('p', float, 3),
                                  ('m', float, 3),
                                  ('avg', float, 3),
                                  ('corr', float),
                                  ('curl', float, 3) ])
    data = np.empty(np.unique(position_data.i).size,
                        dtype=data_format).view(np.recarray)
    data.i = position_data.i
    data.x = position_data.x
    data.d = metrics.d
    data.p = metrics.p
    data.m = metrics.m
    # delta_x = np.empty(np.unique(position_data.i).size, float, 3)
    delta_x = np.empty((np.unique(position_data.i).size, 3))
    end_x = np.empty((np.unique(position_data.i).size, 3))
    for index in np.unique(position_data.i):
        delta_x[index] = data[index].x[-1] - data[index].x[0]
        end_x[index]=data[index].x[-1]
        # print(delta_x)
    data.corr, data.avg = calculate_correlation(end_x, delta_x)
    data.curl = calculate_curl(end_x, delta_x)
    return data

################################################################################

def correct_drift ( data ):
    for index in np.arange(1,len(data.x[0,:,0])):
        difference = data.x[:,index,:] - data.x[:,index-1,:]
        correction = np.mean(difference, axis=0)
        data.x[:,index:,:] -= correction
        # print(np.linalg.norm(correction))
    return data

################################################################################

def calculate_metrics ( data ):
    cell_ids = [entry['i'] for entry in data]
    metric_format = np.dtype([ ('i', int),
                               ('d', float, 3),
                               ('p', float, 3),
                               ('m', float, 3) ])
    metrics = np.empty(len(cell_ids), dtype=metric_format).view(np.recarray)
    positions = np.array
    for index, cell_id in enumerate(cell_ids):
        
        time_max = len(data[index]['x'])

        
        p_0 = data[index]['x'][0,:]
        p_n = data[index]['x'][-1,:]
        # print(p_0)
        path_length = np.sum(np.linalg.norm(
                            data[index]['x'][1:,:] - \
                            data[index]['x'][:-1,:], axis=1))
        
        delta_x = p_n - p_0
        metrics[index].i = cell_id
        metrics[index].d = delta_x * np.linalg.norm(delta_x)  / (6 * time_max)
        metrics[index].p = delta_x/ path_length
        metrics[index].m = np.linalg.norm(delta_x)**2
        
    return metrics

################################################################################

def calculate_curl ( x_array, F_array ):
    curl = np.zeros((len(x_array),3))
    for i in np.arange(0,len(x_array),1):
        seperations = x_array - x_array[i]
        distances = np.linalg.norm(seperations, axis=1)
        closest = np.argsort(distances)[1:]
        total = np.zeros(3)
        counter = 0
        for cell in closest:
            if counter > 6:
                break
            if np.all((x_array[i]-x_array[cell]) != 0):
                total += finite_curl(x_array[i], x_array[cell],
                                     F_array[i], F_array[cell])
                counter += 1
        curl[i,:] = total / 6
    return curl

################################################################################

def finite_curl ( x1, x2, F1, F2 ):
    return np.array([ (F2[2]-F1[2])/(x2[1]-x1[1]) - \
                            (F2[1]-F1[1])/(x2[2]-x1[2]),
                      (F2[0]-F1[0])/(x2[2]-x1[2]) - \
                            (F2[2]-F1[2])/(x2[0]-x1[0]),
                      (F2[1]-F1[1])/(x2[0]-x1[0]) - \
                            (F2[0]-F1[0])/(x2[1]-x1[1]) ])

################################################################################

def calculate_correlation ( x_array, F_array ):
    correlation = np.zeros(len(x_array))
    average = np.zeros((len(x_array),3))
    for i in np.arange(0,len(x_array),1):
        seperations = x_array - x_array[i]
        distances = np.linalg.norm(seperations, axis=1)
        closest = np.argsort(distances)[1:13]
        average[i,:] = np.average(F_array[closest], axis = 0,
                                  weights = distances[closest])
        correlation[i] = np.mean(np.sum(F_array[i] * F_array[closest],
                                 axis=1))
    correlation = correlation / np.amax(correlation)
    return correlation, average

################################################################################

def plot_tracks ( data, ax, threshold ):
    #colormap = plt.get_cmap('viridis') # 'gist_rainbow'
    #metric = np.linalg.norm(data.p, axis=1)
    #cell_colors = colormap(metric)
    line_segments = Line3DCollection(data.x,
                                     colors=(0, 0.4470, 0.7410),
                                     linestyles = 'solid' )
    line_segments.set_array( np.linalg.norm(data.d, axis=1) )
    ax.add_collection(line_segments)
    last_points = np.vstack([entry[-1] for entry in data.x])
    scatter_plot = ax.scatter(last_points[:, 0], last_points[:, 1], last_points[:, 2], 
                              color=(0, 0.4470, 0.7410))
    all_points = np.concatenate(data.x)
    ax.set_xlim([np.amin(all_points[:, 0]), np.amax(all_points[:, 0])])
    ax.set_ylim([np.amin(all_points[:, 1]), np.amax(all_points[:, 1])])
    ax.set_zlim([np.amin(all_points[:, 2]), np.amax(all_points[:, 2])])

    # Set aspect and box aspect
    ax.set_aspect('auto')
    ax.set_box_aspect((np.amax(all_points[:, 0]) - np.amin(all_points[:, 0]),
                   np.amax(all_points[:, 1]) - np.amin(all_points[:, 1]),
                   np.amax(all_points[:, 2]) - np.amin(all_points[:, 2])))
    #mask = metric > threshold
    #line_segments.set_color(np.where(mask[:,np.newaxis],
    #                                 cell_colors,
    #                                 [[0.25,0.25,0.25,0.15]]))
    #scatter_plot.set_color(np.where(mask[:,np.newaxis],
    #                                cell_colors - [0,0,0,0.5],
    #                                [[0.25,0.25,0.25,0.15]]))

################################################################################

def plot_diffusivity ( data, ax, threshold ):
    colormap = plt.get_cmap('viridis') # 'gist_rainbow'
    plt.set_cmap('viridis')
    metric = np.linalg.norm(data.d, axis=1)
    minval = np.amin(metric)
    maxval = np.amax(metric)
    metric = metric / maxval
    cell_colors = colormap(metric)
    line_segments = Line3DCollection(data.x,
                                     cmap=colormap,
                                     linestyles = 'solid' )
    # line_segments = Line3DCollection(np.stack([data.x[:,0,:],
    #                                            data.x[:,-1,:]],
    #                                     axis=1),
    #                                  cmap=colormap,
    #                                  linestyles = 'solid' )
    line_segments.set_array(metric)
    ax.add_collection(line_segments)
    last_points = np.vstack([entry[-1] for entry in data.x])
    scatter_plot = ax.scatter(last_points[:, 0], last_points[:, 1], last_points[:, 2], 
                              color=(0, 0.4470, 0.7410))
    plt.colorbar(cm.ScalarMappable(norm=Normalize(minval, maxval),
                                   cmap=colormap), ax=ax)
    all_points = np.concatenate(data.x)
    ax.set_xlim([np.amin(all_points[:, 0]), np.amax(all_points[:, 0])])
    ax.set_ylim([np.amin(all_points[:, 1]), np.amax(all_points[:, 1])])
    ax.set_zlim([np.amin(all_points[:, 2]), np.amax(all_points[:, 2])])

    # Set aspect and box aspect
    ax.set_aspect('auto')
    ax.set_box_aspect((np.amax(all_points[:, 0]) - np.amin(all_points[:, 0]),
                   np.amax(all_points[:, 1]) - np.amin(all_points[:, 1]),
                   np.amax(all_points[:, 2]) - np.amin(all_points[:, 2])))
    mask = metric > threshold
    line_segments.set_color(np.where(mask[:,np.newaxis],
                                     cell_colors,
                                     [[0.25,0.25,0.25,0.15]]))
    scatter_plot.set_color(np.where(mask[:,np.newaxis],
                                    cell_colors - [0,0,0,0.5],
                                    [[0.25,0.25,0.25,0.15]]))

################################################################################

def plot_persistence ( data, ax, threshold ):
    colormap = plt.get_cmap('viridis') # 'gist_rainbow'
    plt.set_cmap('viridis')
    metric = np.linalg.norm(data.p, axis=1)
    minval = np.amin(metric)
    maxval = np.amax(metric)
    metric = metric / maxval
    cell_colors = colormap(metric)
    line_segments = Line3DCollection(data.x,
                                     cmap=colormap,
                                     linestyles = 'solid' )
    # line_segments = Line3DCollection(np.stack([data.x[:,0,:],
    #                                            data.x[:,-1,:]],
    #                                     axis=1),
    #                                  cmap=colormap,
    #                                  linestyles = 'solid' )
    line_segments.set_array(metric)
    ax.add_collection(line_segments)
    last_points = np.vstack([entry[-1] for entry in data.x])
    scatter_plot = ax.scatter(last_points[:, 0], last_points[:, 1], last_points[:, 2], 
                              color=(0, 0.4470, 0.7410))
    plt.colorbar(cm.ScalarMappable(norm=Normalize(minval, maxval),
                                   cmap=colormap), ax=ax)
    all_points = np.concatenate(data.x)
    ax.set_xlim([np.amin(all_points[:, 0]), np.amax(all_points[:, 0])])
    ax.set_ylim([np.amin(all_points[:, 1]), np.amax(all_points[:, 1])])
    ax.set_zlim([np.amin(all_points[:, 2]), np.amax(all_points[:, 2])])

    # Set aspect and box aspect
    ax.set_aspect('auto')
    ax.set_box_aspect((np.amax(all_points[:, 0]) - np.amin(all_points[:, 0]),
                   np.amax(all_points[:, 1]) - np.amin(all_points[:, 1]),
                   np.amax(all_points[:, 2]) - np.amin(all_points[:, 2])))
    mask = metric > threshold
    line_segments.set_color(np.where(mask[:,np.newaxis],
                                     cell_colors,
                                     [[0.25,0.25,0.25,0.15]]))
    scatter_plot.set_color(np.where(mask[:,np.newaxis],
                                    cell_colors - [0,0,0,0.5],
                                    [[0.25,0.25,0.25,0.15]]))
    

################################################################################

def plot_correlation ( data, ax, threshold ):
    colormap = plt.get_cmap('viridis') # 'gist_rainbow'
    metric = data.corr
    minval = np.amin(metric)
    maxval = np.amax(metric)
    metric = metric / maxval
    cell_colors = colormap(metric)
    line_segments = Line3DCollection(data.x,
                                     cmap=colormap,
                                     linestyles = 'solid' )
    # line_segments = Line3DCollection(np.stack([data.x[:,0,:],
    #                                            data.x[:,-1,:]],
    #                                     axis=1),
    #                                  cmap=colormap,
    #                                  linestyles = 'solid' )
    line_segments.set_array(metric)
    ax.add_collection(line_segments)
    last_points = np.vstack([entry[-1] for entry in data.x])
    scatter_plot = ax.scatter(last_points[:, 0], last_points[:, 1], last_points[:, 2], 
                              color=(0, 0.4470, 0.7410))
    plt.colorbar(cm.ScalarMappable(norm=Normalize(minval, maxval),
                                   cmap=colormap), ax=ax)
    all_points = np.concatenate(data.x)
    ax.set_xlim([np.amin(all_points[:, 0]), np.amax(all_points[:, 0])])
    ax.set_ylim([np.amin(all_points[:, 1]), np.amax(all_points[:, 1])])
    ax.set_zlim([np.amin(all_points[:, 2]), np.amax(all_points[:, 2])])

    # Set aspect and box aspect
    ax.set_aspect('auto')
    ax.set_box_aspect((np.amax(all_points[:, 0]) - np.amin(all_points[:, 0]),
                   np.amax(all_points[:, 1]) - np.amin(all_points[:, 1]),
                   np.amax(all_points[:, 2]) - np.amin(all_points[:, 2])))
    mask = metric > threshold
    line_segments.set_color(np.where(mask[:,np.newaxis],
                                     cell_colors,
                                     [[0.25,0.25,0.25,0.15]]))
    scatter_plot.set_color(np.where(mask[:,np.newaxis],
                                    cell_colors - [0,0,0,0.5],
                                    [[0.25,0.25,0.25,0.15]]))

################################################################################

def plot_deviation ( data, ax, threshold ):
    first_points = np.vstack([entry[0] for entry in data.x])
    last_points = np.vstack([entry[-1] for entry in data.x])
    colormap = plt.get_cmap('viridis') # 'gist_rainbow' 'brg'
    corrected = np.linalg.norm(last_points - data.avg - first_points,
                                axis=1)
    metric = corrected / np.amax(corrected)
    minval = np.amin(metric)
    maxval = np.amax(metric)
    metric = metric / maxval
    cell_colors = colormap(metric)
    tran_colors = cell_colors.copy()
    tran_colors[:,3] = 0.5
    # line_segments = Line3DCollection(
    #                         np.stack([data.x[:,0,:] + data.avg,
    #                                   data.x[:,-1,:]],
    #                                     axis=1),
    #                                 cmap=colormap,
    #                                 linestyles = 'solid' )

    line_segments = Line3DCollection(
                            np.stack([first_points + data.avg,
                                      last_points],
                                        axis=1),
                                    cmap=colormap,
                                    linestyles = 'solid' )
    line_segments.set_array(metric)
    ax.add_collection(line_segments)
    scatter_plot = ax.scatter(last_points[:, 0], last_points[:, 1], last_points[:, 2], 
                              color=(0, 0.4470, 0.7410))
    plt.colorbar(cm.ScalarMappable(norm=Normalize(minval, maxval),
                                   cmap=colormap), ax=ax)
    all_points = np.concatenate(data.x)
    ax.set_xlim([np.amin(all_points[:, 0]), np.amax(all_points[:, 0])])
    ax.set_ylim([np.amin(all_points[:, 1]), np.amax(all_points[:, 1])])
    ax.set_zlim([np.amin(all_points[:, 2]), np.amax(all_points[:, 2])])

    # Set aspect and box aspect
    ax.set_aspect('auto')
    ax.set_box_aspect((np.amax(all_points[:, 0]) - np.amin(all_points[:, 0]),
                   np.amax(all_points[:, 1]) - np.amin(all_points[:, 1]),
                   np.amax(all_points[:, 2]) - np.amin(all_points[:, 2])))
    mask = metric > threshold
    line_segments.set_color(np.where(mask[:,np.newaxis],
                                     cell_colors,
                                     [[0.25,0.25,0.25,0.15]]))
    scatter_plot.set_color(np.where(mask[:,np.newaxis],
                                    cell_colors - [0,0,0,0.5],
                                    [[0.25,0.25,0.25,0.15]]))
                                    
################################################################################

def plot_msd ( data, ax, threshold ):
    colormap = plt.get_cmap('viridis') # 'gist_rainbow'
    plt.set_cmap('viridis')
    metric = np.linalg.norm(data.m, axis=1)
    minval = np.amin(metric)
    maxval = np.amax(metric)
    metric = metric / maxval
    cell_colors = colormap(metric)
    line_segments = Line3DCollection(data.x,
                                     cmap=colormap,
                                     linestyles = 'solid' )
    # line_segments = Line3DCollection(np.stack([data.x[:,0,:],
    #                                            data.x[:,-1,:]],
    #                                     axis=1),
    #                                  cmap=colormap,
    #                                  linestyles = 'solid' )
    line_segments.set_array(metric)
    ax.add_collection(line_segments)
    last_points = np.vstack([entry[-1] for entry in data.x])
    scatter_plot = ax.scatter(last_points[:, 0], last_points[:, 1], last_points[:, 2], 
                              color=(0, 0.4470, 0.7410))
    plt.colorbar(cm.ScalarMappable(norm=Normalize(minval, maxval),
                                   cmap=colormap), ax=ax)
    all_points = np.concatenate(data.x)
    ax.set_xlim([np.amin(all_points[:, 0]), np.amax(all_points[:, 0])])
    ax.set_ylim([np.amin(all_points[:, 1]), np.amax(all_points[:, 1])])
    ax.set_zlim([np.amin(all_points[:, 2]), np.amax(all_points[:, 2])])

    # Set aspect and box aspect
    ax.set_aspect('auto')
    ax.set_box_aspect((np.amax(all_points[:, 0]) - np.amin(all_points[:, 0]),
                   np.amax(all_points[:, 1]) - np.amin(all_points[:, 1]),
                   np.amax(all_points[:, 2]) - np.amin(all_points[:, 2])))
    mask = metric > threshold
    line_segments.set_color(np.where(mask[:,np.newaxis],
                                     cell_colors,
                                     [[0.25,0.25,0.25,0.15]]))
    scatter_plot.set_color(np.where(mask[:,np.newaxis],
                                    cell_colors - [0,0,0,0.5],
                                    [[0.25,0.25,0.25,0.15]]))

################################################################################

def setup_textbox (function, layout, label_text,
                   initial_value = 0):
    textbox = QLineEdit()
    need_inner = not isinstance(layout, QHBoxLayout)
    if need_inner:
        inner_layout = QHBoxLayout()
    label = QLabel(label_text)
    label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if need_inner:
        inner_layout.addWidget(label)
    else:
        layout.addWidget(label)
    textbox.setMaxLength(4)
    textbox.setFixedWidth(50)
    textbox.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    textbox.setValidator(QIntValidator())
    textbox.setText(str(initial_value))
    textbox.editingFinished.connect(function)
    if need_inner:
        inner_layout.addWidget(textbox)
        layout.addLayout(inner_layout)
    else:
        layout.addWidget(textbox)
    return textbox

################################################################################

def get_textbox (textbox,
                 minimum_value = None,
                 maximum_value = None):
    value = int(textbox.text())
    if maximum_value is not None:
        if value > maximum_value:
            value = maximum_value
    if minimum_value is not None:
        if value < minimum_value:
            value = minimum_value
    textbox.setText(str(value))
    return value

################################################################################

class MPLCanvas(FigureCanvas):
    def __init__ (self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def reset_ax (self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111, projection='3d')
    

################################################################################

class Window(QWidget):
    def __init__ (self):
        super().__init__()
        self.title = 'Cell Migration Analysis'
        self.datafile = None
        self.data = None
        self.threshold = 0
        self.plot_type = 'None'
        self.canvas = MPLCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.setup_GUI()
    
    def setup_GUI (self):
        self.setWindowTitle(self.title)
        outer_layout = QVBoxLayout()
        plot_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.slider_select)
        plot_layout.addWidget(self.slider)
        mpl_layout = QVBoxLayout()
        mpl_layout.addWidget(self.canvas)
        mpl_layout.addWidget(self.toolbar)
        plot_layout.addLayout(mpl_layout)
        upper_layout = QHBoxLayout()
        self.button_open_file = QPushButton()
        self.button_open_file.setText('Open File')
        self.button_open_file.clicked.connect(self.open_file)
        upper_layout.addWidget(self.button_open_file)
        self.label_file_name = QLabel('No file loaded.')
        self.label_file_name.setAlignment(Qt.AlignCenter)
        upper_layout.addWidget(self.label_file_name)

        self.textbox_t_min = setup_textbox(
                            self.bound_textbox_select,
                            upper_layout, 'Time Min:')
        self.textbox_t_max = setup_textbox(
                            self.bound_textbox_select,
                            upper_layout, 'Time Max:')
        self.reset_time = QPushButton()
        self.reset_time.setText('Reset time')
        self.reset_time.clicked.connect(self.reset_time_bounds)
        upper_layout.addWidget(self.reset_time)
        lower_layout = QHBoxLayout()
        self.button_cell_tracks = QPushButton()
        self.button_cell_tracks.setText('Cell Tracks')
        self.button_cell_tracks.clicked.connect(self.cell_tracks_button)
        lower_layout.addWidget(self.button_cell_tracks)
        self.button_persistence = QPushButton()
        self.button_persistence.setText('Persistence')
        self.button_persistence.clicked.connect(self.persistence_button)
        lower_layout.addWidget(self.button_persistence)
        self.button_diffusivity = QPushButton()
        self.button_diffusivity.setText('Diffusivity')
        self.button_diffusivity.clicked.connect(self.diffusivity_button)
        lower_layout.addWidget(self.button_diffusivity)
        self.button_correlation = QPushButton()
        self.button_correlation.setText('Correlation')
        self.button_correlation.clicked.connect(self.correlation_button)
        lower_layout.addWidget(self.button_correlation)
        self.button_deviation = QPushButton()
        self.button_deviation.setText('Deviation')
        self.button_deviation.clicked.connect(self.deviation_button)
        lower_layout.addWidget(self.button_deviation)
        self.button_msd = QPushButton()
        self.button_msd.setText('MSD')
        self.button_msd.clicked.connect(self.msd_button)
        lower_layout.addWidget(self.button_msd)
        outer_layout.addLayout(plot_layout)
        outer_layout.addLayout(upper_layout)
        outer_layout.addLayout(lower_layout)
        self.setLayout(outer_layout)
    
    def bound_textbox_select (self):
        self.t_min = get_textbox(self.textbox_t_min)
        self.t_max = get_textbox(self.textbox_t_max)
        # print(self.t_min)
        # print(self.t_max)
        return 
    
    def reset_time_bounds (self):
        if self.data is not None:
            self.data = re_process_file(self.datafile, self.t_min, self.t_max)
            print('Time reset')
        return
        
    def open_file (self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self,
                        "Open Cell Migration File",
                        "",
                        "All Files (*);;CSV Files (*.csv);;PKL Files (*.pkl)",
                        options=options)
        if file_name == '':
            return
        self.datafile = Path(file_name)
        #try:
        self.data = process_file(self.datafile)
        self.label_file_name.setText('File: ' + self.datafile.name)
#        except:
#            self.datafile = None
#            msg = QMessageBox()
#            msg.setIcon(QMessageBox.Critical)
#            msg.setText("Error")
#            msg.setInformativeText('Could not open file!')
#            msg.setWindowTitle("Error")
#            msg.exec_()
    
    def slider_select (self):
        if self.data is not None:
            self.threshold = self.slider.value()/100
            self.plot()
    
    def cell_tracks_button (self):
        if self.data is not None:
            self.plot_type = 'Cell Tracks'
            self.plot()
    
    def persistence_button (self):
        if self.data is not None:
            self.plot_type = 'Persistence'
            self.plot()
    
    def diffusivity_button (self):
        if self.data is not None:
            self.plot_type = 'Diffusivity'
            self.plot()
    
    def correlation_button (self):
        if self.data is not None:
            self.plot_type = 'Correlation'
            self.plot()
    
    def deviation_button (self):
        if self.data is not None:
            self.plot_type = 'Deviation'
            self.plot()
    
    def msd_button (self):
        if self.data is not None:
            self.plot_type = 'MSD'
            self.plot()
    
    def plot (self):
        for image in self.canvas.ax.images:
            if image.colorbar is not None:
                image.colorbar.remove()
        for collection in self.canvas.ax.collections:
            if collection.colorbar is not None:
                collection.colorbar.remove()
        self.canvas.ax.clear()
        self.canvas.reset_ax()
        if self.plot_type == 'None':
            return
        elif self.plot_type == 'Cell Tracks':
            plot_tracks(self.data, self.canvas.ax, self.threshold)
        elif self.plot_type == 'Persistence':
            plot_persistence(self.data, self.canvas.ax, self.threshold)
        elif self.plot_type == 'Diffusivity':
            plot_diffusivity(self.data, self.canvas.ax, self.threshold)
        elif self.plot_type == 'Correlation':
            plot_correlation(self.data, self.canvas.ax, self.threshold)
        elif self.plot_type == 'Deviation':
            plot_deviation(self.data, self.canvas.ax, self.threshold)
        elif self.plot_type == 'MSD':
            plot_msd(self.data, self.canvas.ax, self.threshold)
        else:
            return
        self.canvas.draw()


################################################################################

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

################################################################################
# EOF

# to do 1. add detected t_min and t_max. 2. Toggle at current view. 3. Colormap reset and label.