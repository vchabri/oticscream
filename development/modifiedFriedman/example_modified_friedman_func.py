#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2024

@authors: Vincent Chabridon, Joseph Muré, Elias Fekhari
"""

# %%
import openturns as ot
from oticscream import Icscream
from openturns.viewer import View
import numpy as np

# %%
input_dimension = 20
inputs = ["X" + str(x) for x in range(1, input_dimension + 1)]
output = ["Y"]
formula = "var a[5] := {5.0, 20.0, 8.0, 5.0, 1.5};"
formula += "var X[15] := {" + ",".join(inputs[:15]) + "};"
formula += "var sumsquares :=0;"
formula += "for (var i :=6; i<=15; i+=1) {sumsquares += i^2};"
formula += "var sumrest :=0;"
formula += "for (var i :=6; i<=15; i+=1) {sumrest += i*(X[i-1] - 0.5)};"
formula += "var rest := a[4]/sqrt(sumsquares) * sqrt(12)*sumrest;"
formula += "var sinterm := a[0] * sin(6*pi_*X1^(5/2)*(X2 - 0.5));"
formula += "Y := sinterm + a[1]*(X3 - 0.5)^2 + a[2]*X4 + a[3]*X5 + rest;"
modifiedFriedman = ot.SymbolicFunction(inputs, output, formula)
X = [0.0] * input_dimension
print(modifiedFriedman(X))

# %% Compute reference 1D mean effects

ot.RandomGenerator.SetSeed(0)

distribution_list = [ot.Uniform(0.0, 1.0) for i in range(input_dimension)]
input_distribution = ot.JointDistribution(distribution_list)

input_randomvector = ot.RandomVector(input_distribution)
# output_randomvector = ot.CompositeRandomVector(modifiedFriedman,input_randomvector)

n_sample_reference = 10**6
input_sample = input_randomvector.getSample(n_sample_reference)
output_sample = modifiedFriedman(input_sample)

output_mean = output_sample.computeMean()
print(output_mean)


class ConditionalMeanReference(ot.OpenTURNSPythonFunction):
    def __init__(self, index):
        super().__init__(1, 1)
        super().setInputDescription(["X{}".format(index + 1)])
        super().setOutputDescription(["Y"])
        self._index = index

    def _exec(self, x):
        xx = ot.Sample(input_sample)
        xx[:, self._index] = ot.Sample(xx.getSize(), x)
        return modifiedFriedman(xx).computeMean()


conditional_mean_ref_X1 = ot.Function(ConditionalMeanReference(0))
conditional_mean_ref_X2 = ot.Function(ConditionalMeanReference(1))
conditional_mean_ref_X3 = ot.Function(ConditionalMeanReference(2))
conditional_mean_ref_X4 = ot.Function(ConditionalMeanReference(3))
conditional_mean_ref_X5 = ot.Function(ConditionalMeanReference(4))


# %%
graph = conditional_mean_ref_X1.draw(0.0, 1.0)
line_X1 = graph.getDrawable(0)

# %%
graph = conditional_mean_ref_X2.draw(0.0, 1.0)
line_X2 = graph.getDrawable(0)


# %%
graph = conditional_mean_ref_X3.draw(0.0, 1.0)
line_X3 = graph.getDrawable(0)


# %%
graph = conditional_mean_ref_X3.draw(0.0, 1.0)
line_X3 = graph.getDrawable(0)

# %%
graph = conditional_mean_ref_X4.draw(0.0, 1.0)
line_X4 = graph.getDrawable(0)

# %%
graph = conditional_mean_ref_X5.draw(0.0, 1.0)
line_X5 = graph.getDrawable(0)


# %%
graph.setDrawables([line_X1, line_X2, line_X3, line_X4, line_X5])
graph.setLegends(["X1", "X2", "X3", "X4", "X5"])
graph.setLegendPosition("bottomright")
graph.setTitle("")
graph.setXTitle("")
# %%
View(graph)


# %%
class ConditionalExceedanceProbability(ot.OpenTURNSPythonFunction):
    def __init__(self, index, quantile=0.9):
        super().__init__(1, 1)
        super().setInputDescription(["X{}".format(index + 1)])
        super().setOutputDescription(["Y"])
        self._index = index
        self._threshold = output_sample.computeQuantile(quantile)[0]

    def _exec(self, x):
        xx = ot.Sample(input_sample)
        xx[:, self._index] = ot.Sample(xx.getSize(), x)
        outputs = np.array(modifiedFriedman(xx))
        return ot.Point(1, np.mean(outputs > self._threshold))


conditional_proba_ref_X1 = ot.Function(ConditionalExceedanceProbability(0))
conditional_proba_ref_X2 = ot.Function(ConditionalExceedanceProbability(1))
conditional_proba_ref_X3 = ot.Function(ConditionalExceedanceProbability(2))
conditional_proba_ref_X4 = ot.Function(ConditionalExceedanceProbability(3))
conditional_proba_ref_X5 = ot.Function(ConditionalExceedanceProbability(4))

# %%
graph = conditional_proba_ref_X1.draw(0.0, 1.0)
line_X1 = graph.getDrawable(0)

# %%
graph = conditional_proba_ref_X2.draw(0.0, 1.0)
line_X2 = graph.getDrawable(0)


# %%
graph = conditional_proba_ref_X3.draw(0.0, 1.0)
line_X3 = graph.getDrawable(0)


# %%
graph = conditional_proba_ref_X3.draw(0.0, 1.0)
line_X3 = graph.getDrawable(0)

# %%
graph = conditional_proba_ref_X4.draw(0.0, 1.0)
line_X4 = graph.getDrawable(0)

# %%
graph = conditional_proba_ref_X5.draw(0.0, 1.0)
line_X5 = graph.getDrawable(0)
# %%
graph.setDrawables([line_X1, line_X2, line_X3, line_X4, line_X5])
graph.setLegends(["X1", "X2", "X3", "X4", "X5"])
graph.setLegendPosition("bottomright")
graph.setTitle("")
graph.setXTitle("")
# %%
