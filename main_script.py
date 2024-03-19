#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2023

@author: Vincent Chabridon
"""
# %%
import os

work_path = "/home/g44079/Documents/01_PROJETS/THCOEURS/oticscream/oticscream"
os.chdir(work_path)
import oticscream as oti
import pandas as pd
import openturns as ot
import numpy as np
import time as tm

# %%
# G-Sobol function
# ---------------------------
ot.RandomGenerator.SetSeed(0)

d_scenario = 3
d_rand = 5
d = d_rand + d_scenario

distributionList = [ot.Uniform(0.0, 1.0) for i in range(d)]
myDistribution = ot.ComposedDistribution(distributionList)

# %%
def GSobolModel(X):
    X = ot.Point(X)
    d = X.getDimension()
    Y = 1.0
    a = ot.Point(d)
    for i in range(d):
        a[i] = i
        Y *= (abs(4.0 * X[i] - 2.0) + a[i]) / (1.0 + a[i])  # product
    return ot.Point([Y])


myModel = ot.PythonFunction(d, 1, GSobolModel)

variable_names = ["X" + str(x) for x in range(1, d + 1)]

Nsim = 300
myinputsample = myDistribution.getSample(Nsim)
myinputsample.setDescription(variable_names)
myoutputsample = myModel(myinputsample)
myoutputsample.setDescription(["Y"])

dfx = myinputsample.asDataFrame()
dfy = myoutputsample.asDataFrame()

df = pd.concat([dfx, dfy.reindex(dfx.index)], axis=1)

myfigpath = "figures_gsobol/"

# %%
# Build lists of scenario variables and random variables
# ---------------------------
variables_names = df.columns.tolist()
list_inputs = variables_names[:-1]
list_scenario_variables = ["X1", "X2"]
list_random_variables = [x for x in list_inputs if x not in list_scenario_variables]

# %%
icscream = oti.Icscream(
    dataset=df,
    scenario_variables_columns=list_scenario_variables,
    random_variables_columns=list_random_variables,
    output_variable_column=["Y"],
    figpath=myfigpath,
)

t_0 = tm.time()
icscream.draw_output_sample_analysis()
icscream.draw_sensitivity_results()
icscream.aggregate_pvalues_and_sort_variables()
icscream.build_and_validate_kriging_metamodel(
    nugget_factor=1e-6, optimization_algo="LN_COBYLA", nsample_multistart=10
)

marginals_X_Tilda = [ot.Uniform(0.0, 1.0) for i in range(len(icscream.X_Tilda))]
composed_distribution_X_Tilda = ot.ComposedDistribution(marginals_X_Tilda)

icscream.compute_conditional_probabilities(
    composed_distribution_X_Tilda, n_sample_X_Tilda=200
)
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec / 60),
    "(min)",
)

# %%
# Modified Friedman function
# ---------------------------
ot.RandomGenerator.SetSeed(0)

d_scenario = 5
d_rand = 15
d = d_rand + d_scenario

distributionList = [ot.Uniform(0.0, 1.0) for i in range(d)]
myDistribution = ot.ComposedDistribution(distributionList)


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
myinputsample = myDistribution.getSample(Nsim)
myinputsample.setDescription(variable_names)
myoutputsample = myModel(myinputsample)
myoutputsample.setDescription(["Y"])

dfx = myinputsample.asDataFrame()
dfy = myoutputsample.asDataFrame()

df = pd.concat([dfx, dfy.reindex(dfx.index)], axis=1)

# Build lists of scenario variables and random variables
# ---------------------------
variables_names = df.columns.tolist()
list_inputs = variables_names[:-1]
list_scenario_variables = ["X1", "X2", "X3", "X4", "X5"]
list_random_variables = [x for x in list_inputs if x not in list_scenario_variables]

myfigpath = "figures_friedman/"

icscream = oti.Icscream(
    dataset=df,
    scenario_variables_columns=list_scenario_variables,
    random_variables_columns=list_random_variables,
    output_variable_column=["Y"],
    figpath=myfigpath,
)

t_0 = tm.time()
icscream.draw_output_sample_analysis()
icscream.draw_sensitivity_results()
icscream.aggregate_pvalues_and_sort_variables()
icscream.build_and_validate_kriging_metamodel(
    nugget_factor=1e-6, optimization_algo="LN_COBYLA", nsample_multistart=5
)

marginals_X_Tilda = [ot.Uniform(0.0, 1.0) for i in range(len(icscream.X_Tilda))]
composed_distribution_X_Tilda = ot.ComposedDistribution(marginals_X_Tilda)

icscream.compute_conditional_probabilities(
    composed_distribution_X_Tilda, n_sample_X_Tilda=100
)
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec / 60),
    "(min)",
)

# %%
# ROSA2
# ---------------------------
dpath = "data_files/"

# Import dataset as a pd.DataFrame
# ---------------------------
df_full = pd.read_csv(dpath + "ROSA2_data_withcolnames.csv")
# data.setDescription(['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','Y'])
n_select = 500
df = df_full.sample(n_select)
df = df.reset_index(drop=True)

# Input dimension and sample size
# ---------------------------
d = df.shape[1] - 1
print(">> Input dimension =", d)
print(">> Sample size =", n_select)

# Build lists of scenario variables and random variables
# ---------------------------
variables_names = df.columns.tolist()
list_inputs = variables_names[:-1]
list_scenario_variables = ["X1", "X2", "X3", "X4", "X5"]
list_random_variables = [x for x in list_inputs if x not in list_scenario_variables]
# list_bounds_scenario_variables = [(0,1),(0,1),(0,1),(0,1),(0,1)]

# oticscream on the ROSA2 dataset
# ---------------------------
myfigpath = "figures_rosa2/"

icscream = oti.Icscream(
    dataset=df,
    scenario_variables_columns=list_scenario_variables,
    random_variables_columns=list_random_variables,
    output_variable_column=["Y"],
    figpath=myfigpath,
)

t_0 = tm.time()
icscream.draw_output_sample_analysis()
icscream.draw_sensitivity_results()
icscream.aggregate_pvalues_and_sort_variables()
icscream.build_and_validate_kriging_metamodel(
    nugget_factor=1e-6, optimization_algo="LN_COBYLA", nsample_multistart=5
)

# marginals_X_Tilda = [ot.Uniform(0.0, 1.0) for i in range(len(icscream.X_Tilda))]
# composed_distribution_X_Tilda = ot.ComposedDistribution(marginals_X_Tilda)

# icscream.compute_conditional_probabilities(composed_distribution_X_Tilda, n_sample_X_Tilda=100)
time_elapsed_sec = tm.time() - t_0
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec),
    "(sec)",
)
print(
    ">> Info: Elapsed time for the whole run:",
    "{:.6}".format(time_elapsed_sec / 60),
    "(min)",
)
