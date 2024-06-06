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
condmean_X1 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X1")
condmean_X2 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X2")
condmean_X3 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X3")
condmean_X4 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X4")
condmean_X5 = icscream_7.build_1D_conditional_mean_as_PythonFunction("X5")
# %%
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

# %%
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
for num, line in enumerate(graph_proba.getDrawables()):
    line.setLineStyle("dashed")
    line.setColor(line.getColor())
    line.setLegend("")
    graph_proba_analytical.add(line)
graph_proba_analytical.setTitle("Analytical vs GP conditional exceedance probability")
v = View(graph_proba_analytical)
v.save("figures/Conditional_proba_GPvsAnalytical.pdf")

# %%
