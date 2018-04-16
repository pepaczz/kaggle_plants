# -*- coding: utf-8 -*-
# Copyright (C) 2018, Josef Brechler. 
# Follows Kaggle Competitions license, see https://www.kaggle.com/terms
# Some codes used in class plants_data_preprocessing were adopted
#    from https://www.kaggle.com/gaborvecsei/plants-t-sne submission
#

import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np

import pickle 
import os.path
import warnings

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output


def visualize_scatter_with_images(X_2d_data, images, figsize=(45,45), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()
    
def visualize_scatter_corrected(data_2d, label_ids, id_to_label_dict, figsize=(20,20), render_image=True):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    
    if not render_image:
        return plt
    

def generate_tsne_result_filename(p, lr, nit, plant_ids):
    '''
    Generate filename from parameters (perplexity, learning rate) and ids of plants.
    
    :param p: perplexity
    :param lt: learning rate
    :param plant_ids: ids of plants to be used
    :return: string with filename (with .obj extension, without path)
    '''
    # generate plants code
    plants_code = ''.join('p'+str(i) for i in plant_ids)

    # generate perplexity code
    per_code = 'per'+str(p)

    # generate learning rate code
    lr_code = 'lr'+str(lr)
    
    # generate iterations code
    nit_code = 'nit'+str(nit)

    # join everyhting together
    return '_'.join(code for code in ('tsne', plants_code, per_code, lr_code, nit_code)) + '.obj'

def extract_plants_ids(fname):
    '''
    Extract ids of plants from a filename. Currently not used.
    Regex resources e.g. at https://docs.python.org/3/howto/regex.html
    
    Example usage: extract_plants_ids('tsne_p12p10_per10_lr50.obj')
    
    :param fname: string witi filename
    :return: list of plants' ids (integers)
    '''
    import re

    # get the flowers code
    plants_code = re.search('\_p\d+[a-zA-Z0-9]*\_', my_string).group()

    # get flowers ids
    plants_ids_str = re.findall('[0-9]+', plants_code)
    plants_ids = list(map(int, plants_ids_str))
    plants_ids.sort()
    
    return plants_ids  
    
        
def pickle_wrapper(mode, fname, x=None, rewrite=False):
    '''
    read or write an object into a pickle file
    '''

    if mode == 'w':
        
        # test for existence of file if necessary
        if not rewrite:
            if os.path.isfile(fname):
                warnings.warn("File already exists", Warning)
                return None
                
        # write into a file
        filehandler = open(fname, 'wb') 
        pickle.dump(x, filehandler)
        filehandler.close()
        
        return None

    if mode == 'r':

        # read from file
        filehandler = open(fname,'rb')
        x = pickle.load(filehandler) 
        filehandler.close()
        return x