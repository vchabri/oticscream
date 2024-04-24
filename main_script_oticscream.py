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

# %%
# Modified Friedman function (ref Marrel et al. 2021)
# ---------------------------
ot.RandomGenerator.SetSeed(0)

d_penalized = 5
d_aleatory = 15
d = d_aleatory + d_penalized

distributionList = [ot.Uniform(0.0, 1.0) for i in range(d)]
myDistribution = ot.JointDistribution(distributionList)


def ModifiedFriedmanModel(X):
    X = ot.Point(X)
    d = X.getDimension()
    Y = 1.0
    a = [5.0, 20.0, 8.0, 5.0, 1.5]
    Ii = np.arange(len(a) + 1, d - len(a))
    b = np.sum(Ii**2)
    X_ = np.array(X)[Ii.astype(int)]
    # Bug OpenTURNS: cannot access ot.Sample() elements with a list of np.int64
    c = np.sum(np.sqrt(12) * Ii * (X_ - 0.5))
    r = (a[4] / np.sqrt(b)) * c
    Y = (
        a[0] * np.sin(6 * np.pi * (X[0] ** (5.0 / 2.0)) * (X[1] - 0.5))
        + a[1] * (X[2] - 0.5) ** 2
        + a[2] * X[3]
        + a[3] * X[4]
        + r
    )
    return ot.Point([Y])


myModel = ot.PythonFunction(d, 1, ModifiedFriedmanModel)

variable_names = ["X" + str(x) for x in range(1, d + 1)]

Nsim = 500
input_sample = myDistribution.getSample(Nsim)
input_sample.setDescription(variable_names)
output_sample = myModel(input_sample)
output_sample.setDescription(["Y"])

scaler = StandardScaler()
transformed_input_sample = scaler.fit_transform(input_sample)

dfx = pd.DataFrame(transformed_input_sample)
dfx.columns = list(input_sample.getDescription())
dfy = output_sample.asDataFrame()

list_penalized_variables = ["X1", "X2", "X3", "X4", "X5"]
list_aleatory_variables = [x for x in dfx.columns if x not in list_penalized_variables]

df_penalized = dfx[list_penalized_variables]
df_aleatory = dfx[list_aleatory_variables]

dist_penalized = myDistribution.getMarginal(range(d_penalized))
dist_aleatory = myDistribution.getMarginal(range(d_penalized,d))

icscream = Icscream(df_penalized=df_penalized,
                    df_aleatory=df_aleatory,
                    dist_penalized=dist_penalized,
                    dist_aleatory=dist_aleatory,
                    df_output=dfy)

#------------------#
# t_0 = tm.time()
# #icscream.draw_output_sample_analysis()
# icscream.perform_TSA_study()
# time_elapsed_sec = tm.time() - t_0
# print(
#     ">> Info: Elapsed time for the whole run:",
#     "{:.6}".format(time_elapsed_sec),
#     "(sec)",
# )
# #------------------#
# t_0 = tm.time()
# icscream.perform_CSA_study()
# time_elapsed_sec = tm.time() - t_0
# print(
#     ">> Info: Elapsed time for the whole run:",
#     "{:.6}".format(time_elapsed_sec),
#     "(sec)",
# )
#------------------#
t_0 = tm.time()

results_sa = icscream.draw_sensitivity_results()

time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
#------------------#

# %%
t_0 = tm.time()

results_aggreg = icscream.aggregate_pvalues_and_sort_variables()

time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)

# %%
