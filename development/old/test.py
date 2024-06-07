#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2024

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
myDistribution = ot.JointDistribution(distributionList)  ## ot.ComposedDistribution(distributionList)

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
# %%
df1 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])

df2 = pd.DataFrame(np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
                   columns=['d', 'e', 'f'])
# %%
