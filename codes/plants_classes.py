# -*- coding: utf-8 -*-
# Copyright (C) 2018, Josef Brechler. 
# Follows Kaggle Competitions license, see https://www.kaggle.com/terms
# Some codes used in class plants_data_preprocessing were adopted
#    from https://www.kaggle.com/gaborvecsei/plants-t-sne submission
#

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils

import pickle 
import os.path

import pandas as pd

from math import sqrt

from helper_functions import *
from plants_functions import *


class plants_data_preprocessing(object):
    '''
    Class for data preprocessing
    
     - Method: __init__(self, train_data_fld,prepr_img_fname, prepr_lab_fname, 
                        prepr_data_fld, lab_enum_fname, force_preprocess=False)
     - Method: preprocess(self)
     - Method: preprocess_for_cnn(self)
     - Method: plot_plants(self, plant_ids, img_index=0, img_per_row=6)
    '''
    
    def __init__(self, train_data_fld,prepr_img_fname, 
                 prepr_lab_fname, prepr_data_fld, lab_enum_fname, force_preprocess=False):
        '''
        Initialize and decide whether data will be preprocessed or loaded from binaries
        '''
        
        self.train_data_fld = train_data_fld
        self.prepr_img_fname = prepr_img_fname
        self.prepr_lab_fname = prepr_lab_fname
        self.prepr_data_fld = prepr_data_fld
        self.lab_enum_fname = lab_enum_fname
        self.force_preprocess = force_preprocess
        
        # check for existence of files
        labels_enumerators_file_exist = py_in([self.lab_enum_fname],os.listdir(self.prepr_data_fld))[0]
        images_file_exist = py_in([self.prepr_img_fname],os.listdir(self.prepr_data_fld))[0]
        labels_file_exist = py_in([self.prepr_lab_fname],os.listdir(self.prepr_data_fld))[0]
        
        # condition whether data will be preprocesses or loaded
        self.preprocess_data = not (labels_enumerators_file_exist and 
                                    images_file_exist and 
                                    labels_file_exist and 
                                    (not force_preprocess))
        
    def preprocess(self):
        '''
        Either load training data, preprocess them and save into binaries. Or load binaries.
        '''
        
        # preprocess images and labels only if appropriate files do not exist
        if self.preprocess_data:

            images = []
            labels = []

            # iterate over all subfolders in train data folder
            for i, class_folder_name in enumerate(os.listdir(self.train_data_fld)): 

                print('folder %s / %s' % (i+1, len(os.listdir(self.train_data_fld))))
                # merge paths
                class_folder_path = os.path.join(self.train_data_fld, class_folder_name)
                # iterate over each image
                for image_path in glob(os.path.join(class_folder_path, "*.png")):

                    # read the image
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                    # resize it to 150x150
                    image = cv2.resize(image, (150, 150))
                    image = segment_plant(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                    image = cv2.resize(image, (45,45))
                    image = cv2.resize(image, (65,65))

                    image = image.flatten()

                    images.append(image)
                    labels.append(class_folder_name)

            images = np.array(images)
            labels = np.array(labels)
            
            # define dictionaries for conversion between ids and names of plants
            label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
            id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
            self.labels_enumerators = {'label_to_id_dict':label_to_id_dict,
                                       'id_to_label_dict':id_to_label_dict}
            
            # convert labels to ids
            self.label_ids = np.array([label_to_id_dict[x] for x in labels])
            
            # rescale images
            self.images_scaled = StandardScaler().fit_transform(images)

            # write into binaries
            # ?note: rescaled images take much more memory to save (73 MB vs 9 MB)
            pickle_wrapper(mode='w',
                           fname=os.path.join(self.prepr_data_fld, self.lab_enum_fname),
                           x = self.labels_enumerators,
                           rewrite=True)
            
            pickle_wrapper(mode='w',
                           fname=os.path.join(self.prepr_data_fld, self.prepr_img_fname),
                           x = self.images_scaled,
                           rewrite=True)

            pickle_wrapper(mode='w',
                           fname=os.path.join(self.prepr_data_fld, self.prepr_lab_fname),
                           x = self.label_ids,
                           rewrite=True)

        else:
            # load from binaries
            print('Loading preprocessed image data from binaries')
            self.labels_enumerators = pickle_wrapper(mode='r',
                                                fname=os.path.join(self.prepr_data_fld, self.lab_enum_fname))
            self.images_scaled = pickle_wrapper(mode='r',
                                                fname=os.path.join(self.prepr_data_fld, self.prepr_img_fname))
            self.label_ids = pickle_wrapper(mode='r',
                                            fname=os.path.join(self.prepr_data_fld, self.prepr_lab_fname))

    def preprocess_for_cnn(self):
        '''
        Do some more preprocessin needed for CNN
        (reshape, rescale)
        '''
        
        
        # rescale X to [0,1] - before only unit variance
        X=self.images_scaled
        for ind, row in enumerate(X):
            X[ind,:]=( (X[ind,:] - np.min(X[ind,:]))/(np.max(X[ind,:] - np.min(X[ind,:])))) 
        
        # de-flatten X - assumes square shape
        img_size=int(sqrt(X[0].shape[0]))
        self.X_for_cnn = np.reshape(X, (len(X), img_size, img_size, 1))
        
        # label-binarize Y
        num_classes = len(self.labels_enumerators['id_to_label_dict'])
        Y = np_utils.to_categorical(self.label_ids, num_classes)
        self.Y_for_cnn = Y
        
        
    
            
    def plot_plants(self, plant_ids, img_index=0, img_per_row=6):
        '''
        Plot panel of images containing both the original and preprocessed images

        :param plant_ids:
        :param img_index:
        :param img_per_row:
        '''

        # define figure
        nrow = (((len(plant_ids) - 1) // img_per_row) + 1) * 2
        fig = plt.figure(figsize=(15,nrow*2.5))

        # dictionary to convert between labels and ids
        id_to_label_dict = self.labels_enumerators['id_to_label_dict']

        # iterate over plant_ids
        for i, plant_id in enumerate(plant_ids):

            ### get original image
            # convert id to labels
            plant_label = id_to_label_dict[plant_id]
            # print(plant_label)

            # get list of files for the plant
            class_folder_path = os.path.join(self.train_data_fld, plant_label)
            image_path = glob(os.path.join(class_folder_path, "*.png"))[img_index]

            # read the image
            image_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_orig_resized = cv2.resize(image_orig, (150, 150))

            ### get preprocessed image
            # get index of image within data
            plant_index = py_which(py_in(self.label_ids, [plant_id]))[img_index]

            # get and resize the image
            image_preprocessed_resized = np.reshape(self.images_scaled[plant_index], (45,45))

            ### plot
            # calculate positions within subplot
            pos_orig = ((i) % 6 + 1) + ((i) // img_per_row) * img_per_row * 2
            pos_rescaled = ((i) % 6 + 1) + ((i) // img_per_row) * img_per_row * 2 + img_per_row

            # original image
            plt.subplot(nrow, img_per_row, pos_orig) 
            plt.imshow(image_orig_resized)
            plt.title(plant_label + ' (id ' + str(plant_id)  + ')')
            plt.axis('off')

            # preprocessed image
            plt.subplot(nrow, img_per_row, pos_rescaled)
            plt.imshow(image_preprocessed_resized, cmap="gray")
            plt.axis('off')

class plants_dim_reduction(object):
    '''
    Class that takes preprocessed data and applies PCA ans t-SNE functions
    
    Method: __init__(self, data_complete, label_ids_complete, plant_ids, params=None)
    Method: fit_pca(self, n_components=180)
    Method: plot_pca_explained_variance(self)
    Method: fit_tsne(self, tsne_results_folder, recalculate_all=False, 
                     save_into_binary=True, verbose=1, n_iter = 1500)
    Method: plot_tsne_scatter(self, id_to_label_dict, figsize=(15, 12))
    
    '''
    
    def __init__(self, data_complete, label_ids_complete, plant_ids, params=None):
        
        self.plant_ids = plant_ids
        
        # save default parameters if not supplied
        if params is None:
            self.params = {'perplexity_grid':[10,50,100], 'learning_rate_grid':[50,200,500]}
        else:
            self.params = params

        # get indices of selected plants from plant_ids
        selected_row_indices = py_which(py_in(label_ids_complete, self.plant_ids))   

        # use these indices to select appropriate rows from data
        self.data = data_complete[selected_row_indices, :]
        self.label_ids = label_ids_complete[selected_row_indices]  

        self.param_grid = [] # declare param grid
        plant_ids.sort() # sort plant ids

        # iterate over all combinations of parameters
        for p in self.params['perplexity_grid']:
            for lr in self.params['learning_rate_grid']:

                # print('perplexity: %s, learning rate: %s' % (p, lr))
                param_dict_now = {'perplexity':p, 'learning_rate':lr}  # write the parameters
                self.param_grid.append(param_dict_now)   # apend to results
        
        
    def fit_pca(self, n_components=180):
        '''
        Fit PCA
        
        :param n_components:
        '''
        
        # declare and fit the PCA object
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(self.data)

        # calculate the explained variance ratio and cumulative evr
        self.pca_evr = pca.explained_variance_ratio_ 
        self.pca_evr_cs = self.pca_evr.cumsum()
    
    
    def plot_pca_explained_variance(self):
        '''
        Plot graph of cumulative explained variance
        TO DO: add render_plot=False parameter to return the image itself
        
        :param render_plot:
        :return:
        '''

        # prepare df for plotting
        df_pca_evr_cs = pd.DataFrame({'explained_var':self.pca_evr_cs})
        df_pca_evr_cs = df_pca_evr_cs.reset_index()

        plt.plot(df_pca_evr_cs.explained_var)

    
    def fit_tsne(self, tsne_results_folder, recalculate_all=False, save_into_binary=True, verbose=1, n_iter = 1500):
        '''
        Fit t-SNE and save results into binary
        
        :param tsne_results_folder:
        :param recalculate_all:
        :param save_into_binary:
        :param verbose:
        :param n_iter:
        :return:
        '''
        
        # generate all filenames and save into param_grid
        for i in range(len(self.param_grid)):
            
            fname = generate_tsne_result_filename(p=self.param_grid[i]['perplexity'],
                                                  lr=self.param_grid[i]['learning_rate'],
                                                  nit=n_iter,
                                                  plant_ids=self.plant_ids)
            self.param_grid[i]['fname'] = fname
        
        # names of binaries of existing results
        fnames_existing = os.listdir(tsne_results_folder)

        # names of files to be calculated
        fnames_grid = [x['fname'] for x in self.param_grid]

        # intersection and complement (calculated and needed to be calculated)
        indices_done = py_which(py_in(fnames_grid, fnames_existing))
        indices_undone = py_which(py_not(py_in(fnames_grid, fnames_existing)))
        indices_all = list(range(len(self.param_grid)))

        # prepare indices for what needs to be calculated
        if recalculate_all:
            self.indices_to_calculate = indices_all # use indices of all elements
        else:
            self.indices_to_calculate = indices_undone # use only what is not calculated

        # prepare indices for what needs to be loaded
        self.indices_to_load = py_which(py_not(py_in(indices_all, self.indices_to_calculate)))

        # tsne results declare
        self.tsne_results = [None] * len(self.param_grid)

        ### calculation part

        # iterate over all param cobinations necessary to calculate
        for n, i in enumerate(self.indices_to_calculate):

            print('calculating index:'+str(i)+', '+str(n+1)+'/'+str(len(self.indices_to_calculate)))

            # declare tsne object
            tsne = TSNE(n_components=2, 
                        perplexity=self.param_grid[i]['perplexity'],
                        learning_rate=self.param_grid[i]['learning_rate'],
                        verbose=verbose, 
                        n_iter=n_iter)
            tsne_fitted = tsne.fit_transform(self.pca_result)

            # save the results
            self.tsne_results[i] = tsne_fitted

            # save into binary if required
            if save_into_binary:
                filehandler = open(os.path.join(tsne_results_folder, self.param_grid[i]['fname']), 'wb') 
                pickle.dump(tsne_fitted, filehandler)
                filehandler.close()

        ### loading part
        
        # print progress
        if (verbose >= 1) and (len(self.indices_to_load) > 0):
            print('loading ' + str(len(self.indices_to_load)) + ' t-NSE results')
        
        for n, i in enumerate(self.indices_to_load):
            #print('loading index:'+str(i)+', '+str(n)+'/'+str(len(self.indices_to_load)))

            filehandler = open(os.path.join(tsne_results_folder, self.param_grid[i]['fname']), 'rb') 
            self.tsne_results[i] = pickle.load(filehandler) 
            filehandler.close()
            
        ###
        
        # rescale t-SNE results
        self.tsne_results_scaled = [StandardScaler().fit_transform(res) for res in self.tsne_results]
    
    
    def plot_tsne_scatter(self, id_to_label_dict, figsize=(15, 12)):
        '''
        Visualize t-SNE results in scatter plot
        
        :param id_to_label_dict: Conversion between ids and plants names. Likely
                            data_preprocessing.labels_enumerators['id_to_label_dict']
        :param figsize: size of figure
        '''

        fig = plt.figure(figsize=figsize)

        # iterate over each of the parameters combinations
        for i in range(len(self.tsne_results_scaled)):

            data_2d = self.tsne_results_scaled[i]
            plt.subplot(3, 3, i+1)

            n_classes = len(np.unique(self.label_ids)) # number of plants ids

            # iterate over every plant id
            for j, label_id in enumerate(np.unique(self.label_ids)):
                plt.scatter(data_2d[np.where(self.label_ids == label_id), 0],
                            data_2d[np.where(self.label_ids == label_id), 1],
                            s=3,
                            marker='o',
                            color= plt.cm.Set1(j / float(n_classes)),
                            linewidth='1',
                            alpha=0.6,
                            label=id_to_label_dict[label_id])

            plt.title('perplexity: ' + str(self.param_grid[i]['perplexity']) + 
                      ', learning_rate: ' + str(self.param_grid[i]['learning_rate']))
            
            # plot only single legend
            if i == 1:
                # set legend labels alpha
                plt.legend(loc='best')
                leg = plt.legend()
                for lh in leg.legendHandles: 
                    lh.set_alpha(1)
            
            # plt.close()

    