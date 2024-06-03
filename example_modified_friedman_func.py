#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2024

@authors: Vincent Chabridon, Joseph Muré, Elias Fekhari
"""

#%%
import openturns as ot
from oticscream import Icscream
from openturns.viewer import View
from copy import deepcopy

# %%
input_dimension = 20
inputs = ['X' + str(x) for x in range(1, input_dimension + 1)]
output = ['Y']
formula = 'var a[5] := {5.0, 20.0, 8.0, 5.0, 1.5};'
formula += 'var X[15] := {X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15};'
formula += 'var sumsquares :=0;'
formula += 'for (var i :=6; i<=15; i+=1) {sumsquares += i^2};'
formula += 'var sumrest :=0;'
formula += 'for (var i :=6; i<=15; i+=1) {sumrest += i*(X[i-1] - 0.5)};'
formula += 'var rest := a[4]/sqrt(sumsquares) * sqrt(12)*sumrest;'
formula += 'var sinterm := a[0] * sin(6*pi_*X1^(5/2)*(X2 - 0.5));'
formula += 'Y := sinterm + a[1]*(X3 - 0.5)^2 + a[2]*X4 + a[3]*X5 + rest;'
modifiedFriedman = ot.SymbolicFunction(inputs, output, formula)
X = [0.0]*input_dimension
print(modifiedFriedman(X))

# %% Compute reference 1D mean effects

ot.RandomGenerator.SetSeed(0)

distribution_list = [ot.Uniform(0.0, 1.0) for i in range(input_dimension)]
input_distribution = ot.JointDistribution(distribution_list)

input_randomvector = ot.RandomVector(input_distribution)
#output_randomvector = ot.CompositeRandomVector(modifiedFriedman,input_randomvector)

n_sample_reference = 10**6
input_sample = input_randomvector.getSample(n_sample_reference)
output_sample = modifiedFriedman(input_sample)

output_mean = output_sample.computeMean()
#print(output_mean)

#frozenX1_function = ot.ParametricFunction(modifiedFriedman,[0],[0.5])

def conditional_mean_reference_X1(x):
    xx = deepcopy(input_sample) #### WARNING ::: EXTREMELY SLOOOOWWWW (Idea: use np. vstack / hstack instead)
    xx[:,0] = ot.Sample(xx.getSize(),x)

    return modifiedFriedman(xx).computeMean()

conditional_mean_ref_X1_PF = ot.PythonFunction(1,1, conditional_mean_reference_X1)

# print(conditional_mean_reference_X1([0.4]))

# conditional_mean_ref_X1_PF.draw(0.0,1.0)


# %%
