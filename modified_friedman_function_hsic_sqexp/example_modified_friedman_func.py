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
# Define the modified Friedman function

input_dimension = 20
inputs = ["X" + str(x) for x in range(1, input_dimension + 1)]
output = ["Y"]
formula = "var a[5] := {5.0, 20.0, 8.0, 5.0, 1.5};"
formula += "var X[15] := {" + ",".join(inputs[:15]) + "};"
formula += "var sumsquares :=0;"
formula += "for (var i :=6; i<=15; i+=1) {sumsquares += (15-i+1)^2};"
formula += "var sumrest :=0;"
formula += "for (var i :=6; i<=15; i+=1) {sumrest += (15-i+1)*(X[i-1] - 0.5)};"
formula += "var rest := a[4]/sqrt(sumsquares) * sqrt(12)*sumrest;"
formula += "var sinterm := a[0] * sin(6*pi_*X1^(5/2)*(X2 - 0.5));"
formula += "Y := sinterm + a[1]*(X3 - 0.5)^2 + a[2]*X4 + a[3]*X5 + rest;"
modifiedFriedman = ot.SymbolicFunction(inputs, output, formula)
X = [0.0] * input_dimension
print(modifiedFriedman(X))

# %%
# Generate the input and output samples

distribution_list = [ot.Uniform(0.0, 1.0) for i in range(input_dimension)]
input_distribution = ot.JointDistribution(distribution_list)
input_description = [
    "X{}".format(i) for i in range(1, input_distribution.getDimension() + 1)
]
input_distribution.setDescription(input_description)

input_randomvector = ot.RandomVector(input_distribution)
# output_randomvector = ot.CompositeRandomVector(modifiedFriedman,input_randomvector)

n_sample_reference = 10 ** 6
input_sample = input_randomvector.getSample(n_sample_reference)
output_sample = modifiedFriedman(input_sample)

output_mean = output_sample.computeMean()[0]
print(output_mean)


# %% Compute reference 1D mean effects


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
# Compute reference mean effect wrt all 5 penalized input variables


def conditional_mean_ref_penalized(X):
    xx = ot.Sample(input_sample)
    xx[:, 0:5] = ot.Sample(xx.getSize(), X)
    return modifiedFriedman(xx).computeMean()


conditional_mean_ref_penalized = ot.PythonFunction(5, 1, conditional_mean_ref_penalized)


# %%
# Compute reference exceedance probability with one fixed input variable


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


# %% BUILD ALL PENALIZED 2D COND EXCEEDANCE PROBABILITY


class ConditionalExceedanceProbability2D(ot.OpenTURNSPythonFunction):
    def __init__(self, index1, index2, quantile=0.9):
        super().__init__(2, 1)
        super().setInputDescription(["X%i" % (index1 + 1), "X%i" % (index2 + 1)])
        super().setOutputDescription(["Y"])
        self._index1 = index1
        self._index2 = index2
        self._threshold = output_sample.computeQuantile(quantile)[0]

    def _exec(self, x):
        xx = ot.Sample(input_sample)
        xx[:, [self._index1, self._index2]] = ot.Sample(xx.getSize(), x)
        outputs = np.array(modifiedFriedman(xx))
        return ot.Point(1, np.mean(outputs > self._threshold))


conditional_proba_ref_X1_X2 = ot.Function(ConditionalExceedanceProbability2D(0, 1))

