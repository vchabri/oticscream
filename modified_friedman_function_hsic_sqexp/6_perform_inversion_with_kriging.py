#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2024

@authors: Vincent Chabridon, Joseph Muré, Elias Fekhari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
from sklearn.preprocessing import StandardScaler
import openturns as ot
from oticscream import Icscream

ot.RandomGenerator.SetSeed(0)

mydummysample = ot.Sample([[1],[2],[3],[4]]).asDataFrame()

icscream_6 = Icscream(df_penalized=mydummysample,
                      df_aleatory=mydummysample,
                      dist_penalized=None,
                      dist_aleatory=None,
                      df_output=mydummysample)

icscream_6.load("save_files/save_after_kriging_validation.npy")

# %%
#------------------#
t_0 = tm.time()

res_X_tilda_given_data = icscream_6.construct_and_sample_x_tilda_distribution()
# print(res_X_tilda_given_data)

res_X_penalized_given_data  = icscream_6.construct_and_sample_x_penalized_distribution()
# print(res_X_penalized_given_data)

icscream_6.create_full_sample_for_metamodel_prediction()
print(icscream_6._full_sample_variable_names)
print(icscream_6._full_sample)
print(icscream_6._X_Penalized_indices_within_full_sample)

time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
#------------------#
#%%
icscream_6.save("save_files/save_test_X_tilda_X_pen_given_data.npy")

#%%
# from openturns.viewer import View

# graph = ot.VisualTest.DrawPairs(res_X_penalized_given_data)
# view = View(graph)
# plt.show()

