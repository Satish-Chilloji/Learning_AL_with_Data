import numpy as np
import pickle as pkl
import scipy
import matplotlib.pyplot as plt

# from Dataset4LAL import DatasetSimulated
# from Tree4LAL import Tree4LAL
# from LALmodel import LALmodel

experiment = dict()
# number of datasets for which we will generate data
experiment['n_datasets'] = 500
# how many datapoints will be labelled at the beginning, including 1 positive and 1 negative
experiment['n_labelleds'] = np.arange(2,50,1)
# how many times we will sample data with the same parameters
experiment['n_points_per_experiment'] = 10
# dimensionality of the data
experiment['n_dim'] = 2
# measure of quality change
experiment['method'] = 'error'
# for now 2 techniques for tree growing are available, random that means just adding random samples and iterative for adding points based on previously build model
experiment['treegrowing'] = 'iterative'