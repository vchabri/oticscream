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

icscream_3 = Icscream(df_penalized=mydummysample,
                      df_aleatory=mydummysample,
                      dist_penalized=None,
                      dist_aleatory=None,
                      df_output=mydummysample)

icscream_3.load("save_files/modFriedmanfunc_afteraggregation.npy")

# %%
#------------------#
t_0 = tm.time()
results_build_explanatory = icscream_3.build_explanatory_variables()
icscream_3.setup_trend_and_covariance_models(trend_factory="LinearBasisFactory") #"ConstantBasisFactory"
icscream_3.build_kriging_data()
icscream_3.build_train_and_validation_sets_by_greedy_support_points()
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
#------------------#
#%%
icscream_3.save("save_files/save_afteric3.npy")