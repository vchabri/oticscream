#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2024

@authors: Vincent Chabridon, Joseph Muré, Elias Fekhari
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import openturns as ot
from oticscream import Icscream
from openturns.viewer import View

ot.RandomGenerator.SetSeed(0)

mydummysample = ot.Sample([[1], [2], [3], [4]]).asDataFrame()

icscream_7 = Icscream(
    df_penalized=mydummysample,
    df_aleatory=mydummysample,
    dist_penalized=None,
    dist_aleatory=None,
    df_output=mydummysample,
)

icscream_7.load("save_files/save_test_X_tilda_X_pen_given_data.npy")

# %%
# Validate the GP surrogate model

import example_modified_friedman_func as fr

# %%
kriging_inputs = fr.input_sample.getMarginal(
    icscream_7._kriging_metamodel.getInputDescription()
)
# %%
kriging_output = icscream_7._kriging_metamodel(kriging_inputs)
# %%
validation = ot.MetaModelValidation(fr.output_sample, kriging_output)
# %%
# Q2 score

q2 = validation.computeR2Score()[0]
print(q2)  # expect 0.795667
# %%
# Graphical validation

graph = validation.drawValidation().getGraph(0, 0)
graph.setTitle("Q2 coefficient: %.2f" % q2)
v = View(graph, scatter_kw={"alpha": 0.05, "marker": "*"})
# %%
kriging_result = icscream_7._kriging_result
# %%
variance = kriging_result.getConditionalMarginalVariance(kriging_inputs)
# %%
std = np.sqrt(variance)


# %%
def coverage(alpha):
    bound = ot.DistFunc.qNormal(0.5 + alpha[0] / 2) * std
    error = kriging_output - fr.output_sample
    correct = np.abs(error) < bound
    return [np.mean(correct)]


# %%
coverage = ot.PythonFunction(1, 1, coverage)
coverage.setInputDescription(["alpha"])
coverage.setOutputDescription(["coverage probability"])
# %%
low = 0.0
graph = coverage.draw(low, 1.0)
graph.add(ot.Curve([low, 1.0], [low, 1.0]))
graph.setLegendPosition("topleft")
graph.setLegends(["coverage", ""])
View(graph)
# %%
low = 0.8
graph = coverage.draw(low, 1.0)
graph.add(ot.Curve([low, 1.0], [low, 1.0]))
graph.setLegendPosition("topleft")
graph.setLegends(["coverage", ""])  # %%
View(graph)

# %%
# Use ICSCREAM to build the 1D conditional means from the GP

condmean_X1 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X1")
condmean_X2 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X2")
condmean_X3 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X3")
condmean_X4 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X4")
condmean_X5 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X5")
# %%
# 1D conditional means from the GP

graph = condmean_X1.draw(0.0, 1.0, 100)
graph.add(condmean_X2.draw(0.0, 1.0, 100))
graph.add(condmean_X3.draw(0.0, 1.0, 100))
graph.add(condmean_X4.draw(0.0, 1.0, 100))
graph.add(condmean_X5.draw(0.0, 1.0, 100))
mean = icscream_7.compute_mean()
mean_line = ot.Curve([0.0, 1.0], [mean, mean])
mean_line.setColor("black")
graph.add(mean_line)
graph.setLegends(["X%d" % i for i in range(1, 6)] + ["global mean"])
graph.setLegendPosition("bottomright")
graph.setTitle("GP conditional mean")
graph.setXTitle("Input value")
v = View(graph)
v.save("figures/Conditional_mean_GP.pdf")

# %%
# 1D conditional means from the reference function

graph_analytical = fr.conditional_mean_ref_X1.draw(0.0, 1.0)
graph_analytical.add(fr.conditional_mean_ref_X2.draw(0.0, 1.0))
graph_analytical.add(fr.conditional_mean_ref_X3.draw(0.0, 1.0))
graph_analytical.add(fr.conditional_mean_ref_X4.draw(0.0, 1.0))
graph_analytical.add(fr.conditional_mean_ref_X5.draw(0.0, 1.0))
mean_line = ot.Curve([0.0, 1.0], [fr.output_mean, fr.output_mean])
mean_line.setColor("black")
graph_analytical.add(mean_line)
graph_analytical.setLegends(["X1", "X2", "X3", "X4", "X5", "global mean"])
graph_analytical.setLegendPosition("bottomright")
graph_analytical.setTitle("Analytical conditional mean")
graph_analytical.setXTitle("Input value")
v = View(graph_analytical)
v.save("figures/Conditional_mean_Analytical.pdf")
# %%
# Overlay 1D conditional means from the reference function and the GP

for num, line in enumerate(graph.getDrawables()):
    line.setLineStyle("dashed")
    graph.setDrawable(line, num)
graph.setColors(graph.getColors())  # to make them no longer count as "default"
graph_analytical.add(graph)
graph_analytical.setLegends(
    ["X%i" % i for i in range(1, 6)] + ["global mean"] + [""] * 6
)
graph_analytical.setTitle("Analytical vs GP conditional mean")
v = View(graph_analytical)
v.save("figures/Conditional_mean_GPvsAnalytical.pdf")

# %%
# Use ICSCREAM to build the 1D conditional exceedance probabilities from the GP

condprob_X1 = icscream_7.build_1D_conditional_exceedance_probability_as_PythonFunction(
    "X1"
)
condprob_X2 = icscream_7.build_1D_conditional_exceedance_probability_as_PythonFunction(
    "X2"
)
condprob_X3 = icscream_7.build_1D_conditional_exceedance_probability_as_PythonFunction(
    "X3"
)
condprob_X4 = icscream_7.build_1D_conditional_exceedance_probability_as_PythonFunction(
    "X4"
)
condprob_X5 = icscream_7.build_1D_conditional_exceedance_probability_as_PythonFunction(
    "X5"
)

# %%
# 1D conditional exceednance probabilities from the GP

graph_proba = condprob_X1.draw(0.0, 1.0, 100)
graph_proba.add(condprob_X2.draw(0.0, 1.0, 100))
graph_proba.add(condprob_X3.draw(0.0, 1.0, 100))
graph_proba.add(condprob_X4.draw(0.0, 1.0, 100))
graph_proba.add(condprob_X5.draw(0.0, 1.0, 100))
proba = icscream_7.compute_exceedance_probability()
proba_line = ot.Curve([0.0, 1.0], [proba, proba])
proba_line.setColor("black")
graph_proba.add(proba_line)
graph_proba.setLegends(["X1", "X2", "X3", "X4", "X5", "unconditional"])
graph_proba.setLegendPosition("top")
graph_proba.setTitle("GP conditional exceedance probability")
graph_proba.setXTitle("Input value")
v = View(graph_proba)
v.save("figures/Conditional_proba_GP.pdf")
# %%
# 1D conditional exceednance probabilities from the reference function

graph_proba_analytical = fr.conditional_proba_ref_X1.draw(0.0, 1.0)
graph_proba_analytical.add(fr.conditional_proba_ref_X2.draw(0.0, 1.0))
graph_proba_analytical.add(fr.conditional_proba_ref_X3.draw(0.0, 1.0))
graph_proba_analytical.add(fr.conditional_proba_ref_X4.draw(0.0, 1.0))
graph_proba_analytical.add(fr.conditional_proba_ref_X5.draw(0.0, 1.0))
proba = 0.1
proba_line = ot.Curve([0.0, 1.0], [proba, proba])
proba_line.setColor("black")
graph_proba_analytical.add(proba_line)
graph_proba_analytical.setLegends(["X1", "X2", "X3", "X4", "X5", "unconditional"])
graph_proba_analytical.setLegendPosition("top")
graph_proba_analytical.setTitle("Analytical conditional exceedance probability")
graph_proba_analytical.setXTitle("Input value")
v = View(graph_proba_analytical)
v.save("figures/Conditional_proba_Analytical.pdf")

# %%
# Overlay 1D conditional exceednance probabilities
# from the reference function and the GP

for num, line in enumerate(graph_proba.getDrawables()):
    line.setLineStyle("dashed")
    line.setColor(line.getColor())
    line.setLegend("")
    graph_proba_analytical.add(line)
graph_proba_analytical.setTitle("Analytical vs GP conditional exceedance probability")
v = View(graph_proba_analytical)
v.save("figures/Conditional_proba_GPvsAnalytical.pdf")


# %%
# REFERENCE FUNCTION CONDITIONAL EXCEEDANCE PROBABILITY wrt 2 FIXED INPUTS

ot.ResourceMap.SetAsBool("Contour-DefaultIsFilled", True)
ot.ResourceMap.SetAsUnsignedInteger("Contour-DefaultLevelsNumber", 20)

dim = 5

lowerBound = ot.Point(dim, 0.0)
upperBound = ot.Point(dim, 1.0)

vmin = np.inf
vmax = -np.inf

grid = ot.GridLayout(dim - 1, dim - 1)
grid.setTitle("Modified Friedman exceedance probability with 2 fixed input variables")
for i in range(1, dim):
    for j in range(i):
        # Definition of 2D conditional mean function
        crossCutFunction = ot.Function(fr.ConditionalExceedanceProbability2D(i, j))
        crossCutLowerBound = [lowerBound[j], lowerBound[i]]
        crossCutUpperBound = [upperBound[j], upperBound[i]]

        # Get and customize the contour ploticscream_7._X_Penalized[i]
        graph = crossCutFunction.draw(crossCutLowerBound, crossCutUpperBound, [5, 5])
        graph.setTitle("")
        contour = graph.getDrawable(0).getImplementation()
        vmin = min(vmin, contour.getData().getMin()[0])
        vmax = max(vmax, contour.getData().getMax()[0])
        contour.setColorBarPosition("")  # suppress colorbar of each plot
        contour.setColorMap("viridis")
        graph.setDrawable(contour, 0)
        graph.setXTitle("")
        graph.setYTitle("")
        graph.setTickLocation(ot.GraphImplementation.TICKNONE)
        graph.setGrid(False)

        # Creation of axes title
        if j == 0:
            graph.setYTitle("X%i" % (i + 1))
        if i == dim - 1:
            graph.setXTitle("X%i" % (j + 1))

        grid.setGraph(i - 1, j, graph)

for i in range(1, dim):
    for j in range(i):
        graph = grid.getGraph(i - 1, j)
        contour = graph.getDrawable(0).getImplementation()
        contour.setVmin(vmin)
        contour.setVmax(vmax)
        graph.setDrawable(contour, 0)
        grid.setGraph(i - 1, j, graph)

# Get View object to manipulate the underlying figure
v = View(grid)
fig = v.getFigure()
fig.set_size_inches(12, 12)  # reduce the size


# Setup a large colorbar
axes = v.getAxes()
colorbar = fig.colorbar(
    v.getSubviews()[2][2].getContourSets()[0], ax=axes[:, -1], fraction=0.1
)

fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
fig.savefig("Friedman_exceedance_proba_2fixed.pdf")

# %%
# OPTIMIZATION

# %%
# Optimize the conditional mean wrt all penalized variables
# i.e. X1, X2, X3, X4, X5

mean_GP_conditional_to_penalized = (
    icscream_7.build_allpenalized_conditional_mean_as_PythonFunction()
)
# %%
problem = ot.OptimizationProblem(mean_GP_conditional_to_penalized)
problem.setMinimization(False)
bounds = ot.Interval([0.0] * 5, [1.0] * 5)
problem.setBounds(bounds)
optim = ot.NLopt("LN_COBYLA")
optim.setProblem(problem)

bounded_dist_multi = ot.JointDistribution([ot.Uniform(0.0, 1.0)] * 5)
start = ot.LowDiscrepancyExperiment(
    ot.SobolSequence(), bounded_dist_multi, 10, True
).generate()

multistart = ot.MultiStart(optim, start)
# %%
multistart.run()
res = multistart.getResult()
print(res.getOptimalPoint())

# %%
# Optimize the exceedance probability wrt all penalized variables
# i.e. X1, X2, X3, X4, X5

mean_GP_proba_to_penalized = (
    icscream_7.build_allpenalized_conditional_exceedance_probability_as_PythonFunction()
)
# %%
problem_proba = ot.OptimizationProblem(mean_GP_proba_to_penalized)
problem_proba.setMinimization(False)
bounds = ot.Interval([0.0] * 5, [1.0] * 5)
problem_proba.setBounds(bounds)
optim_proba = ot.NLopt("LN_COBYLA")
optim_proba.setProblem(problem_proba)

bounded_dist_multi = ot.JointDistribution([ot.Uniform(0.0, 1.0)] * 5)
start = ot.LowDiscrepancyExperiment(
    ot.SobolSequence(), bounded_dist_multi, 10, True
).generate()

multistart_proba = ot.MultiStart(optim_proba, start)

# %%
multistart_proba.run()
res_proba = multistart_proba.getResult()
print(res_proba.getOptimalPoint())
