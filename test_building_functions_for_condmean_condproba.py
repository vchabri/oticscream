#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2024

@authors: Vincent Chabridon, Joseph Muré, Elias Fekhari
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
from sklearn.preprocessing import StandardScaler
import openturns as ot
from oticscream import Icscream
from openturns.viewer import View

ot.RandomGenerator.SetSeed(0)

mydummysample = ot.Sample([[1],[2],[3],[4]]).asDataFrame()

icscream_7 = Icscream(df_penalized=mydummysample,
                      df_aleatory=mydummysample,
                      dist_penalized=None,
                      dist_aleatory=None,
                      df_output=mydummysample)

icscream_7.load("save_files/save_test_X_tilda_X_pen_given_data.npy")

# %%
res_1dcondmean = icscream_7.build_1D_conditional_mean("X3")

condmean_1D = icscream_7.build_1D_conditional_mean_as_PythonFunction("X3")
condmean_1D.draw(0.0, 1.0, 100)

#%%
res_2dcondmean = icscream_7.build_2D_conditional_mean("X1","X3")

ot.ResourceMap.SetAsString("Contour-DefaultColorMap", 'viridis')
ot.ResourceMap.SetAsBool("Contour-DefaultIsFilled", True)
condmean_2D = icscream_7.build_2D_conditional_mean_as_PythonFunction("X1","X3")
condmean_2D_memoize = ot.MemoizeFunction(condmean_2D)
condmean_2D_memoize.draw([0.0, 0.0], [1.0, 1.0], [10, 10])
# condmean_2D_memoize.getInputHistory()

# %%
res_allcondmean = icscream_7.build_allpenalized_conditional_mean()

allcondmean = icscream_7.build_allpenalized_conditional_mean_as_PythonFunction()

ot.ResourceMap.SetAsBool("Contour-DefaultIsFilled", True)
ot.ResourceMap.SetAsUnsignedInteger("Contour-DefaultLevelsNumber", 20)

dim = len(icscream_7._X_Penalized)

lowerBound = ot.Point(dim, 0.0)
upperBound = ot.Point(dim, 1.0)


grid = ot.GridLayout(dim - 1, dim - 1)
for i in range(1, dim):
    for j in range(i):
        crossCutIndices = []
        crossCutReferencePoint = []
        for k in range(dim):
            if k != i and k != j:
                crossCutIndices.append(k)
                # Definition of the reference point
                crossCutReferencePoint.append(icscream_7._sample_penalized.computeMean()[k])

        # Definition of 2D cross cut function
        crossCutFunction = ot.ParametricFunction(
            allcondmean, crossCutIndices, crossCutReferencePoint
        )
        crossCutLowerBound = [lowerBound[j], lowerBound[i]]
        crossCutUpperBound = [upperBound[j], upperBound[i]]

        # Get and customize the contour plot
        graph = crossCutFunction.draw(crossCutLowerBound, crossCutUpperBound, [5, 5])
        graph.setTitle("")
        contour = graph.getDrawable(0).getImplementation()
        contour.setVmin(7.0)
        contour.setVmax(15.0)
        contour.setColorBarPosition("")  # suppress colorbar of each plot
        contour.setColorMap("viridis")
        graph.setDrawable(contour, 0)
        graph.setXTitle("")
        graph.setYTitle("")
        graph.setTickLocation(ot.GraphImplementation.TICKNONE)
        graph.setGrid(False)

        # Creation of axes title
        if j == 0:
            graph.setYTitle(icscream_7._X_Penalized[i])
        if i == 9:
            graph.setXTitle(icscream_7._X_Penalized[j])

        grid.setGraph(i - 1, j, graph)

# %%
# Get View object to manipulate the underlying figure
v = View(grid)
fig = v.getFigure()
fig.set_size_inches(12, 12)  # reduce the size

#%%
# Setup a large colorbar
axes = v.getAxes()
colorbar = fig.colorbar(
    v.getSubviews()[0][0].getContourSets()[0], ax=axes[:, -1], fraction=0.1
)

#%%
fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
# %%
