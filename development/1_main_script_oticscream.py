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
from example_modified_friedman_func import modifiedFriedman

# %%
# Modified Friedman function (ref Marrel et al. 2021)
# ---------------------------
ot.RandomGenerator.SetSeed(2)

d_penalized = 5

d_aleatory = 15
d = d_aleatory + d_penalized

distributionList = [ot.Uniform(0.0, 1.0) for i in range(d)]
myDistribution = ot.JointDistribution(distributionList)

variable_names = ["X" + str(x) for x in range(1, d + 1)]

Nsim = 250 # number in Marrel et al.
input_sample = myDistribution.getSample(Nsim)
input_sample.setDescription(variable_names)
output_sample = modifiedFriedman(input_sample)
output_sample.setDescription(["Y"])

# scaler = StandardScaler()
# transformed_input_sample = scaler.fit_transform(input_sample)

# dfx = pd.DataFrame(transformed_input_sample)
dfx = pd.DataFrame(np.array(input_sample))
dfx.columns = list(input_sample.getDescription())
dfy = output_sample.asDataFrame()

list_penalized_variables = ["X1", "X2", "X3", "X4", "X5"]
list_aleatory_variables = [x for x in dfx.columns if x not in list_penalized_variables]

df_penalized = dfx[list_penalized_variables]
df_aleatory = dfx[list_aleatory_variables]

dist_penalized = myDistribution.getMarginal(range(d_penalized))
dist_aleatory = myDistribution.getMarginal(range(d_penalized, d))

icscream = Icscream(
    df_penalized=df_penalized,
    df_aleatory=df_aleatory,
    dist_penalized=dist_penalized,
    dist_aleatory=dist_aleatory,
    df_output=dfy,
)

#%%
icscream.draw_output_sample_analysis()

#%%
# ------------------#
t_0 = tm.time()
icscream.perform_TSA_study()
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
#%%
# ------------------#
t_0 = tm.time()
icscream.perform_CSA_study()
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
# ------------------#
#%%
# ------------------#
t_0 = tm.time()
results_sa = icscream.draw_sensitivity_results()
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
# ------------------#
#%%
icscream.save("save_files/modFriedmanfunc_afterHSICstudy.npy")


# ------------------#
t_0 = tm.time()
results_aggreg = icscream.aggregate_pvalues_and_sort_variables()
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
# ------------------#

print("Primary effects:")
print(icscream._X_Primary_Influential_Inputs)

print("Secondary effects:")
print(icscream._X_Secondary_Influential_Inputs)
#%%

interesting = input_sample.getMarginal(icscream._X_Primary_Influential_Inputs)
interesting = ot.Sample(np.hstack((interesting, output_sample.rank())))
interesting.setDescription(icscream._X_Primary_Influential_Inputs + ["Y"])
# ot.VisualTest.DrawPairs(interesting)
# %%
