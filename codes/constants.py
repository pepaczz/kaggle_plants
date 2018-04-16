# -*- coding: utf-8 -*-
# Copyright (C) 2018, Josef Brechler. 
# Follows Kaggle Competitions license, see https://www.kaggle.com/terms
# Some codes used in class plants_data_preprocessing were adopted
#    from https://www.kaggle.com/gaborvecsei/plants-t-sne submission
#

import os

BASE_DATA_FOLDER = "../input"
TRAIN_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")
PREPROCESSED_DATA_FOLDER = "../saved_objects/preprocessed_data"
TSNE_RESULTS_FOLDER = "../saved_objects/tsne_results"
MODEL_INPUT_DATA_FOLDER = "../saved_objects/model_input_data"

PREPROCESSED_IMAGES_FNAME = 'preprocessed_images.obj'
PREPROCESSED_LABELS_FNAME = 'preprocessed_labels.obj'
LABELS_ENUMARATORS_FNAME = 'labels_enumerators.obj'

TRAINED_MODELS_FOLDER = "../saved_objects/trained_cnn"
MODEL_HISTORY_FOLDER = "../saved_objects/cnn_history"