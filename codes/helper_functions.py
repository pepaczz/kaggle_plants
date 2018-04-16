# -*- coding: utf-8 -*-
# Copyright (C) 2018, Josef Brechler. 
# Follows Kaggle Competitions license, see https://www.kaggle.com/terms
# Some codes used in class plants_data_preprocessing were adopted
#    from https://www.kaggle.com/gaborvecsei/plants-t-sne submission
#

import pandas as pd
import numpy as np

#########################################################
### HELPER FUNCTIONS FOR DATAFRAME AND LISTS HANDLING ###

# function to create sample df
def create_sample_df(nrow=15):
    df = pd.DataFrame(np.random.randint(9,size=(nrow,2)), columns=['one', 'two'])
    df_rows = df.shape[0]
    items_xyz = ['xx', 'yy', 'zz']
    items_ab = ['aa', 'bb']
    df['three'] = np.repeat(items_xyz, (df_rows // len(items_xyz)) + 1)[0:df_rows]
    df['four'] = (items_ab*((df_rows // len(items_ab)) + 1))[0:df_rows]
    df.loc[[0,1,8],'one'] = np.nan
    return(df)

### py_which ############################################  

# return list of indices which are True 
# equivalent to the R function 'which'
def py_which(tf_list):
    return([i for i, x in enumerate(tf_list) if x == True])

### py_in ###############################################

# return t/f vector of the 1st list which are in the 2nd list
# equivalent to %in% operator in R
def py_in(x, in_vector):
    return [elem in in_vector for elem in x]

### py_not ##############################################  

# inverts tf values of a boolean list
def py_not(tf_list):
    return [not i for i in tf_list]

### py_select_by_tf ##################################### 

# selects elements of a vector by boolean list
# returns error if vectors do not have a same length
def py_select_by_tf(x, tf_list):

    if len(x) != len(tf_list):
        raise ValueError("Arrays must have the same size")
    
    indices = tuple(py_which(tf_list))
    return([x[i] for i in indices])

### py_select_by_ind #################################### 

# selects elements of a vector by numerical indices
def py_select_by_ind(x, ind_list):
    return([x[i] for i in ind_list])

### py_sort_unique ###################################### 

# return sorted unique values of a list
def py_sort_unique(x):
    y = x.copy()
    y.sort()
    return list(set(y))

#################################################

# return elements of the 1st list which are (not) in the 2nd list
# equivalent to %which% function in R
def compare_list_elements(list_a, list_b, exclude = False):
    tf_indices = [x in list_b for x in list_a]
    
    if exclude:
        tf_indices = [not i for i in tf_indices]

    return(select_list_elements(list_a, tf_indices))

#################################################

# return names of columns of the 1st data frame which are not in the 2nd data frame
def compare_df_names(df_A, df_B, exclude = True):
    a_nm = df_A.columns.tolist()
    b_nm = df_B.columns.tolist()

    return(compare_list_elements(a_nm, b_nm, exclude))

#Method for finding substrings
# taken from https://www.kaggle.com/rcasellas/ensemble-stacking-with-et-script
# perhaps rewrite to a nicer form
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan
            
###############################################  
    
# imputs missing data by selected method applied to groups by selected columns
# this function is deprecated, see the GeneralImputer in submission_RandomForestClassifier_20180307a.py file
def impute_value(df, col_impute, col_group, method = 'median'):
    # tbd: use also different functions
    
    # deep copy the df because of transform
    df_2 = df.copy()
    
    # Create a groupby object: by_sex_class
    grouped = df_2.groupby(col_group)

    # Write a function that imputes median
    def imputer_median(series):
        return series.fillna(series.median())

#     print('-'*10)
#     print(df_2)

    if method == 'median':
        # impute median
        df_2[col_impute] = grouped[col_impute].transform(imputer_median)
        
#         print('-'*10)
#         print(df)
        
        return(df_2)
    else:  
        return np.nan