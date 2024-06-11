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

icscream_4 = Icscream(df_penalized=mydummysample,
                      df_aleatory=mydummysample,
                      dist_penalized=None,
                      dist_aleatory=None,
                      df_output=mydummysample)

icscream_4.load("save_files/save_afteric3.npy")

# %%
#------------------#
t_0 = tm.time()
res_krigmm = icscream_4.build_and_run_kriging_metamodel()
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
#------------------#
#%%
icscream_4.save("save_files/save_after_kriging_training.npy")