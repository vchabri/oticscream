#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2024

@authors: Vincent Chabridon, Joseph Muré, Elias Fekhari
"""
import os

# work_path = "/home/g44079/Documents/01_PROJETS/THCOEURS/oticscream/oticscream"
# os.chdir(work_path)

import numpy as np
import openturns as ot

print("OpenTURNS version:", ot.__version__)
import openturns.viewer as otv

from scipy.spatial.distance import pdist
import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams["text.usetex"] = True
# from matplotlib import rc, rcParams, stylercParams['text.usetex'] = Truerc('font', **{'family': 'Times'})rc('text', usetex=True)rc('font', size=16)# Set the default text font sizerc('axes', titlesize=20)# Set the axes title font sizerc('axes', labelsize=16)# Set the axes labels font sizerc('xtick', labelsize=14)# Set the font size for x tick labelsrc('ytick', labelsize=16)# Set the font size for y tick labelsrc('legend', fontsize=16)# Set the legend font size`
import time as tm
import pickle

ot.Log.Show(ot.Log.NONE)
import otkerneldesign as otkd

## Idées :
# """
# - Passer un coup de Black et Flake
# - A la prochaine release : grosse accélération des p-val permutation + IC par bootstrap
# - Retourner les pandas dataframes de résultats en Table LaTeX
# - Figures : attention, faire un mécanisme de figpath par défaut
# - TODO : faire un check de dimension sur la covariance_collection
# - Réflexions sur aggrégation des pvalperm vs pvalasymp, en tenant compte d'éventuels IC calculés sur ces p-valeurs
# - Ajouter les p-valeurs par CSA dans la stratégie d'agrégation
# - Faire une analyse par cProfile (paquet Python)
# - Faire un module OT + paquet Pip et Conda
# - Séparer les méthodes d'entrainement et de validation du MM de krigeage
# - TODO : traiter un .CSV comme dictionnaire des distributions des entrées (lois et bornes)
# """


def get_indices(greedysupportpoints, sample):
    """
    When provided a subsample of the candidate set, returns the indices of its points in the candidate set.

    Parameters
    ----------
    sample : 2-d list of float
        A subsample of the candidate set.

    Returns
    -------
    indices : list of int
        Indices of the points of the sample within the candidate set.
    """
    sample = np.array(sample)
    if len(sample.shape) != 2:
        raise ValueError(
            "Not a sample: shape is {} instead of 2.".format(len(sample.shape))
        )
    candidate_array = np.array(
        greedysupportpoints._candidate_set
    )  # convert to numpy array so np.where works
    indices = []
    for sample_index, pt in enumerate(sample):
        index = np.where((candidate_array == pt).prod(axis=1))[0]
        if len(index) != 1:
            raise ValueError(
                "The point {}, with index {} in the sample, is not in the candidate set.".format(
                    pt, sample_index
                )
            )
        indices.extend(index)
    return indices


class Icscream:
    """
    Description TODO.

    Parameters
    ----------
    random_distribution : :class:`openturns.Distribution`
    penalized_distribution : :class:`openturns.Distribution`
    model : function
    dataset : pd.DataFrame
    random_variables_columns : list
    penalized_variables_columns : list
    output_variable_column : list
    output_quantile_order : scalar
    p_value_threshold : scalar
    n_sim : integer

    Notes
    -----
    The vectors of penalized and aleatory variables are assumed to be mutually independent.

    Examples
    --------
    >>> todo
    """

    def __init__(
        self,
        df_penalized=None,
        df_aleatory=None,
        dist_penalized=None,
        dist_aleatory=None,
        df_output=None,
        covariance_collection=None,
        output_quantile_order=0.9,
        p_value_threshold=0.05,
        n_perm=200,
    ):
        if (df_penalized is None) or (df_aleatory is None) or (df_output is None):
            raise ValueError(
                "Please, provide lists corresponding to 'df_penalized'', 'df_aleatory' and 'df_output'."
            )
        self._df_penalized = df_penalized
        self._df_aleatory = df_aleatory

        self._df_input = pd.concat([self._df_penalized, self._df_aleatory], axis=1)
        self._df_output = df_output

        self._sample_penalized = ot.Sample.BuildFromDataFrame(self._df_penalized)
        self._sample_aleatory = ot.Sample.BuildFromDataFrame(self._df_aleatory)
        self._sample_output = ot.Sample.BuildFromDataFrame(self._df_output)
        self._sample_input = ot.Sample.BuildFromDataFrame(self._df_input)
        self._dim_penalized = self._sample_penalized.getDimension()
        self._dim_random = self._sample_aleatory.getDimension()
        self._dim_input = self._sample_input.getDimension()

        ## WARNING: set the component names for the aleatory and penalized distributions
        self._dist_penalized = dist_penalized
        self._dist_aleatory = dist_aleatory

        if dist_penalized is not None:
            self._dist_penalized.setDescription(self._sample_penalized.getDescription())

        if dist_aleatory is not None:
            self._dist_aleatory.setDescription(self._sample_aleatory.getDescription())

        self._output_quantile_order = output_quantile_order
        self._empirical_quantile = self._sample_output.computeQuantile(
            self._output_quantile_order
        )[0]
        self._p_value_threshold = p_value_threshold
        self._n_perm = n_perm

        if covariance_collection is None:
            input_covariance_collection = []
            for i in range(self._dim_input):
                Xi = self._sample_input.getMarginal(i)
                input_covariance = ot.SquaredExponential(1)
                input_covariance.setScale(Xi.computeStandardDeviation())
                input_covariance_collection.append(input_covariance)
            output_covariance = ot.SquaredExponential(1)
            output_covariance.setScale(self._sample_output.computeStandardDeviation())
            self._covariance_collection = input_covariance_collection + [
                output_covariance
            ]
        else:
            self._covariance_collection = covariance_collection

        self._GSA_study = None
        self._TSA_study = None
        self._CSA_study = None

        df_columns = (
            self._df_penalized.columns.tolist() + self._df_aleatory.columns.tolist()
        )

        self._GSA_results = pd.DataFrame(
            [],
            columns=df_columns,
            index=[],
        )
        self._TSA_results = pd.DataFrame(
            [],
            columns=df_columns,
            index=[],
        )
        self._CSA_results = pd.DataFrame(
            [],
            columns=df_columns,
            index=[],
        )
        self._Aggregated_pval_results = pd.DataFrame(
            [],
            columns=df_columns,
            index=[],
        )

        self._X_Primary_Influential_Inputs = None
        self._X_Secondary_Influential_Inputs = None
        self._X_Secondary_Influential_Inputs_after_aggregation = None
        self._X_Epsilon = None
        self._X_Penalized = None
        self._X_Explanatory = None
        self._X_Tilda = None
        self._x_data = None
        self._y_data = None

        self._kriging_trend_basis = None
        self._cov_kriging_model = None
        self._x_learn = None
        self._y_learn = None
        self._x_validation = None
        self._y_validation = None
        self._scaled_x_data = None
        self._input_kriging_sample = None
        self._output_kriging_sample = None
        self._kriging_result = None
        self._kriging_metamodel = None

        self._validation_results = None

        self._sample_X_Tilda = None
        self._sample_X_penalized = None

        self._full_sample = None
        self._full_sample_variable_names = None
        self._X_Penalized_indices_within_full_sample = None

        ## Create a standard normal OpenTURNS object
        self._standard_normal_distribution = ot.Normal()

        self._X_Penalized_data = None
        self._X_Penalized_sample = None
        self._list_probabilities_x_pen = None

    def save(self, filename):
        attribute_names = self.__dict__.keys()  # Get the list of attribute names
        with open(filename, "wb") as f:
            for name in attribute_names:
                ## cf. Issue #2624 on openturns' github
                if "study" in name and getattr(self, name) is not None:
                    pickle.dump(
                        "placeholder", f
                    )  # Handle the case of a "_study_" (TO REMOVE ONCE BUG IS FIXED)
                else:
                    pickle.dump(
                        getattr(self, name), f
                    )  # Save the attributes of the self.__dict__

    def load(self, filename):
        attribute_names = self.__dict__.keys()  # Get the list of attribute names
        with open(filename, "rb") as f:
            for name in attribute_names:
                setattr(
                    self, name, pickle.load(f)
                )  # Load each attribute in the initial order

    def draw_output_sample_analysis(self):
        hist = ot.HistogramFactory().build(self._sample_output)
        graph_output = hist.drawPDF()
        #
        kde = ot.KernelSmoothing()
        fit = kde.build(self._sample_output)
        kde_curve = fit.drawPDF()
        graph_output.add(kde_curve)
        #
        mean_Y = self._sample_output.computeMean()[0]
        mean_line = ot.Curve([[mean_Y], [mean_Y]], [[0.0], [fit.computePDF([mean_Y])]])
        mean_line.setLineWidth(2.0)
        graph_output.add(mean_line)
        #
        median_Y = self._sample_output.computeMedian()[0]
        median_line = ot.Curve(
            [[median_Y], [median_Y]], [[0.0], [fit.computePDF([median_Y])]]
        )
        median_line.setLineWidth(2.0)
        graph_output.add(median_line)
        #
        quantile_line = ot.Curve(
            [[self._empirical_quantile], [self._empirical_quantile]],
            [[0.0], [fit.computePDF([self._empirical_quantile])]],
        )
        quantile_line.setLineWidth(2.0)
        graph_output.add(quantile_line)
        #
        # graph_output.setColors(["dodgerblue3", "darkorange1"])
        graph_output.setLegends(
            ["Histogram", "KDE", "Mean", "Median", "Empirical quantile"]
        )
        graph_output.setLegendPosition("topright")
        graph_output.setTitle(
            "Empirical quantile = {:.6}".format(self._empirical_quantile)
        )
        graph_output.setXTitle("Y")
        graph_output.setYTitle("")

        return graph_output

    def set_permutation_size(self, n_perm):
        self._n_perm = n_perm

    def perform_GSA_study(self, hsic_estimator_type=ot.HSICUStat(), savefile=None):
        self._GSA_study = ot.HSICEstimatorGlobalSensitivity(
            self._covariance_collection,
            self._sample_input,
            self._sample_output,
            hsic_estimator_type,
        )
        self._GSA_study.setPermutationSize(self._n_perm)

        self._GSA_results.loc["HSIC", :] = self._GSA_study.getHSICIndices()
        self._GSA_results.loc["R2-HSIC", :] = self._GSA_study.getR2HSICIndices()
        self._GSA_results.loc[
            "p-values perm", :
        ] = self._GSA_study.getPValuesPermutation()
        self._GSA_results.loc[
            "p-values asymp", :
        ] = self._GSA_study.getPValuesAsymptotic()

        if savefile is not None:
            self._GSA_results.to_csv(savefile, index=True)
        return self._GSA_results

    def perform_TSA_study(self, hsic_estimator_type=ot.HSICUStat(), savefile=None):
        critical_domain = ot.Interval(self._empirical_quantile, float("inf"))
        dist_to_critical_domain = ot.DistanceToDomainFunction(critical_domain)
        smoothing_parameter = 0.1 * self._sample_output.computeStandardDeviation()[0]
        f = ot.SymbolicFunction(["x", "s"], ["exp(-x/s)"])
        phi = ot.ParametricFunction(f, [1], [smoothing_parameter])
        filter_function = ot.ComposedFunction(phi, dist_to_critical_domain)

        self._TSA_study = ot.HSICEstimatorTargetSensitivity(
            self._covariance_collection,
            self._sample_input,
            self._sample_output,
            hsic_estimator_type,
            filter_function,
        )
        self._TSA_study.setPermutationSize(self._n_perm)

        self._TSA_results.loc["T-HSIC", :] = self._TSA_study.getHSICIndices()
        self._TSA_results.loc["T-R2-HSIC", :] = self._TSA_study.getR2HSICIndices()
        self._TSA_results.loc[
            "p-values perm", :
        ] = self._TSA_study.getPValuesPermutation()
        self._TSA_results.loc[
            "p-values asymp", :
        ] = self._TSA_study.getPValuesAsymptotic()

        if savefile is not None:
            self._TSA_results.to_csv(savefile, index=True)
        return self._TSA_results

    def perform_CSA_study(self, savefile=None):
        critical_domain = ot.Interval(self._empirical_quantile, float("inf"))
        dist_to_critical_domain = ot.DistanceToDomainFunction(critical_domain)
        smoothing_parameter = 0.1 * self._sample_output.computeStandardDeviation()[0]
        f = ot.SymbolicFunction(["x", "s"], ["exp(-x/s)"])
        phi = ot.ParametricFunction(f, [1], [smoothing_parameter])
        filter_function = ot.ComposedFunction(phi, dist_to_critical_domain)

        self._CSA_study = ot.HSICEstimatorConditionalSensitivity(
            self._covariance_collection,
            self._sample_input,
            self._sample_output,
            filter_function,
        )
        self._CSA_study.setPermutationSize(self._n_perm)

        self._CSA_results.loc["C-HSIC", :] = self._CSA_study.getHSICIndices()
        self._CSA_results.loc["C-R2-HSIC", :] = self._CSA_study.getR2HSICIndices()
        self._CSA_results.loc[
            "p-values perm", :
        ] = self._CSA_study.getPValuesPermutation()

        if savefile is not None:
            self._CSA_results.to_csv(savefile, index=True)
        return self._CSA_results

    def draw_sensitivity_results(self):
        if self._GSA_study is None:
            _ = self.perform_GSA_study()
        if self._TSA_study is None:
            _ = self.perform_TSA_study()
        if self._CSA_study is None:
            _ = self.perform_CSA_study()

        graph_GSA_indices = self._GSA_study.drawR2HSICIndices()
        graph_GSA_indices.setColors(["darkorange1", "black"])
        graph_GSA_indices.setXTitle("inputs")
        graph_GSA_indices.setYTitle("values of indices")
        graph_GSA_indices.setTitle("GSA study - R2-HSIC indices")

        graph_GSA_pval_asymp = self._GSA_study.drawPValuesAsymptotic()
        g = ot.SymbolicFunction("x", "0.05")
        threshold_pval = g.draw(0, self._dim_input)
        graph_GSA_pval_asymp.add(threshold_pval)
        graph_GSA_pval_asymp.setColors(["dodgerblue3", "black", "red"])
        graph_GSA_pval_asymp.setXTitle("inputs")
        graph_GSA_pval_asymp.setYTitle("p-values")
        graph_GSA_pval_asymp.setTitle("GSA study - Asymptotic p-values")

        graph_GSA_pval_perm = self._GSA_study.drawPValuesPermutation()
        g = ot.SymbolicFunction("x", "0.05")
        threshold_pval = g.draw(0, self._dim_input)
        graph_GSA_pval_perm.add(threshold_pval)
        graph_GSA_pval_perm.setColors(["grey", "black", "red"])
        graph_GSA_pval_perm.setXTitle("inputs")
        graph_GSA_pval_perm.setYTitle("p-values")
        graph_GSA_pval_perm.setTitle(
            "GSA study - Permutation p-values ($n_{perm}$ = %d)" % self._n_perm
        )

        graph_TSA_indices = self._TSA_study.drawR2HSICIndices()
        graph_TSA_indices.setColors(["darkorange1", "black"])
        graph_TSA_indices.setXTitle("inputs")
        graph_TSA_indices.setYTitle("values of indices")
        graph_TSA_indices.setTitle("TSA study - R2-HSIC indices")

        graph_TSA_pval_asymp = self._TSA_study.drawPValuesAsymptotic()
        g = ot.SymbolicFunction("x", "0.05")
        threshold_pval = g.draw(0, self._dim_input)
        graph_TSA_pval_asymp.add(threshold_pval)
        graph_TSA_pval_asymp.setColors(["dodgerblue3", "black", "red"])
        graph_TSA_pval_asymp.setXTitle("inputs")
        graph_TSA_pval_asymp.setYTitle("p-values")
        graph_TSA_pval_asymp.setTitle("TSA study - Asymptotic p-values")

        graph_TSA_pval_perm = self._TSA_study.drawPValuesPermutation()
        g = ot.SymbolicFunction("x", "0.05")
        threshold_pval = g.draw(0, self._dim_input)
        graph_TSA_pval_perm.add(threshold_pval)
        graph_TSA_pval_perm.setColors(["grey", "black", "red"])
        graph_TSA_pval_perm.setXTitle("inputs")
        graph_TSA_pval_perm.setYTitle("p-values")
        graph_TSA_pval_perm.setTitle(
            "TSA study - Permutation p-values ($n_{perm}$ = %d)" % self._n_perm
        )

        graph_CSA_indices = self._CSA_study.drawR2HSICIndices()
        graph_CSA_indices.setColors(["darkorange1", "black"])
        graph_CSA_indices.setXTitle("inputs")
        graph_CSA_indices.setYTitle("values of indices")
        graph_CSA_indices.setTitle("CSA study - R2-HSIC indices")

        graph_CSA_pval_perm = self._CSA_study.drawPValuesPermutation()
        g = ot.SymbolicFunction("x", "0.05")
        threshold_pval = g.draw(0, self._dim_input)
        graph_CSA_pval_perm.add(threshold_pval)
        graph_CSA_pval_perm.setColors(["grey", "black", "red"])
        graph_CSA_pval_perm.setXTitle("inputs")
        graph_CSA_pval_perm.setYTitle("p-values")
        graph_CSA_pval_perm.setTitle(
            "CSA study - Permutation p-values ($n_{perm}$ = %d)" % self._n_perm
        )

        graph_dict = {
            "GSA indices": graph_GSA_indices,
            "GSA pval asymp": graph_GSA_pval_asymp,
            "GSA pval perm": graph_GSA_pval_perm,
            "TSA indices": graph_TSA_indices,
            "TSA pval asymp": graph_TSA_pval_asymp,
            "TSA pval perm": graph_TSA_pval_perm,
            "CSA indices": graph_CSA_indices,
            "CSA pval perm": graph_CSA_pval_perm,
        }

        return graph_dict

    def aggregate_pvalues_and_sort_variables(
        self, isAsymptotic=False, sortby_method="Bonferroni"
    ):
        if self._GSA_study is None:
            _ = self.perform_GSA_study()
        if self._TSA_study is None:
            _ = self.perform_TSA_study()
        if self._CSA_study is None:
            _ = self.perform_CSA_study()

        sortby_method += " p-values"

        if isAsymptotic:
            print(">> Info: Asymptotic regime => asymptotic p-values will be used!")
            self._Aggregated_pval_results.loc[
                "GSA p-values", :
            ] = self._GSA_results.loc["p-values asymp", :]
            self._Aggregated_pval_results.loc[
                "TSA p-values", :
            ] = self._TSA_results.loc["p-values asymp", :]
        else:
            print(
                ">> Info: Non-asymptotic regime => permutation-based p-values will be used!"
            )
            self._Aggregated_pval_results.loc[
                "GSA p-values", :
            ] = self._GSA_results.loc["p-values perm", :]
            self._Aggregated_pval_results.loc[
                "TSA p-values", :
            ] = self._TSA_results.loc["p-values perm", :]
            # TODO : inclure  les p-valeurs CSA dans la stratégie globale
            self._Aggregated_pval_results.loc[
                "CSA p-values", :
            ] = self._CSA_results.loc["p-values perm", :]

        pval_GSA = self._Aggregated_pval_results.loc["GSA p-values", :].values
        pval_TSA = self._Aggregated_pval_results.loc["TSA p-values", :].values

        # Aggregate GSA and TSA results using Bonferroni correction
        # ---------------------------
        self._Aggregated_pval_results.loc["Bonferroni p-values", :] = np.minimum(
            2 * np.minimum(pval_GSA, pval_TSA), 1
        )

        # WARNING: TODO below: create an exception in order to handle the case of null p-values (1/0)

        # # Other advanced statistics used for aggregation
        # # ---------------------------
        # self._Aggregated_pval_results.loc["Fisher p-values", :] = -np.log(
        #     pval_GSA.astype("float64")
        # ) - np.log(pval_TSA.astype("float64"))
        # self._Aggregated_pval_results.loc["Tippet p-values", :] = 1 - np.minimum(
        #     pval_GSA, pval_TSA
        # )

        # for i, colname in enumerate(self._sample_input.getDescription()):
        #     invgamma_dist_GSA = ot.InverseGamma(1 / pval_GSA[i], 1.0)
        #     invgamma_dist_TSA = ot.InverseGamma(1 / pval_TSA[i], 1.0)
        #     self._Aggregated_pval_results.loc[
        #         "InvGamma p-values", colname
        #     ] = invgamma_dist_GSA.computeCDF(
        #         1 - pval_GSA[i]
        #     ) + invgamma_dist_TSA.computeCDF(
        #         1 - pval_TSA[i]
        #     )

        # Variable ranking based on the level of the test (first-kind error)
        # ---------------------------
        aggregated_pval = self._Aggregated_pval_results.loc[sortby_method, :]
        self._X_Primary_Influential_Inputs = aggregated_pval[
            aggregated_pval <= self._p_value_threshold
        ].index.tolist()
        self._X_Secondary_Influential_Inputs = aggregated_pval[
            (aggregated_pval > self._p_value_threshold)
            & (aggregated_pval <= 2 * self._p_value_threshold)
        ].index.tolist()
        self._X_Epsilon = aggregated_pval[
            aggregated_pval > 2 * self._p_value_threshold
        ].index.tolist()

        inputs_dict_sorted_variables = {
            "X_Primary_Influential_Inputs": self._X_Primary_Influential_Inputs,
            "X_Secondary_Influential_Inputs": self._X_Secondary_Influential_Inputs,
            "X_Epsilon": self._X_Epsilon,
        }

        return inputs_dict_sorted_variables

    def build_explanatory_variables(self):

        # Penalized variables
        # ---------------------------
        self._X_Penalized = self._df_penalized.columns.tolist()

        # Explanatory variables
        # ---------------------------
        # Trick: use pd.unique() instead of np.unique() to avoid sorting indices
        self._X_Explanatory = pd.unique(
            np.concatenate(
                (self._X_Primary_Influential_Inputs, self._X_Penalized), axis=None
            ).tolist()
        ).tolist()
        # Check whether there is any duplicate between X_Explanatory and X_Secondary_Influential_Inputs
        self._X_Secondary_Influential_Inputs_after_aggregation = [
            elem
            for elem in self._X_Secondary_Influential_Inputs
            if elem not in self._X_Explanatory
        ]
        self._X_Tilda = [
            x
            for x in self._X_Explanatory
            + self._X_Secondary_Influential_Inputs_after_aggregation
            if x not in self._X_Penalized
        ]

        if len(self._X_Explanatory) > 30:
            raise ValueError("The sequential strategy should be implemented!")
        else:
            print(">> Info: Direct metamodel building strategy will be used!")

        inputs_dict_variables_for_metamodeling = {
            "X_Penalized": self._X_Penalized,
            "X_Explanatory": self._X_Explanatory,
            "X_Secondary_Influential_Inputs_after_aggregation": self._X_Secondary_Influential_Inputs_after_aggregation,
            "X_Tilda": self._X_Tilda,
        }

        return inputs_dict_variables_for_metamodeling

    def setup_trend_and_covariance_models(
        self,
        trend_factory="ConstantBasisFactory",
        marginal_cov_model=ot.MaternModel([1.0], [1.0], 5.0 / 2.0),
    ):

        # Managing input dimensions
        # ---------------------------
        dim_krig = len(self._X_Explanatory) + len(
            self._X_Secondary_Influential_Inputs_after_aggregation
        )
        print(">> dim_krig =", dim_krig)

        # Trend
        # ---------------------------
        self._kriging_trend_basis = eval("ot." + trend_factory + "(dim_krig).build()")

        # Covariance model for the explanatory inputs
        # ---------------------------
        cov_X_Explanatory = ot.ProductCovarianceModel(
            [marginal_cov_model] * len(self._X_Explanatory)
        )

        # Covariance model for the secondary influential inputs
        # ---------------------------
        if not len(
            self._X_Secondary_Influential_Inputs_after_aggregation
        ):  # no secondary influential variables
            self._cov_kriging_model = cov_X_Explanatory
        else:
            cov_X_Secondary_Influential_Inputs = ot.IsotropicCovarianceModel(
                marginal_cov_model,
                len(self._X_Secondary_Influential_Inputs_after_aggregation),
            )

            # Resulting covariance model
            # ---------------------------
            self._cov_kriging_model = ot.ProductCovarianceModel(
                [cov_X_Explanatory, cov_X_Secondary_Influential_Inputs]
            )

        # Activate nugget factor for optimization of the homoscedastic nugget effect
        # ---------------------------
        self._cov_kriging_model.activateNuggetFactor(True)

    def build_kriging_data(self):
        ## WARNING:
        ## It is assumed that the data are already scaled.

        # Loading input and output data
        # ---------------------------
        self._x_data = self._sample_input.getMarginal(
            self._X_Explanatory + self._X_Secondary_Influential_Inputs_after_aggregation
        )
        self._y_data = self._sample_output

        # Scaling input data
        # ---------------------------
        # This scaling step is mandatory due to the modeling of the X_SII through an isotropic covariance model which mixes several heterogenous variables (with possible various ranges of correlation lengths)
        ## Warning: in Numpy, the np.std is biased while in Pandas, pd.std is unbiased by default.
        ## COMMENT: it may be possible not to standardize the date. One should be careful with respect to the final optimization step.
        # self._scaled_x_data = (self._x_data - self._x_data.mean(axis=0)) / self._x_data.std(
        #     axis=0, ddof=1
        # )
        # scaled_x_data_sample = ot.Sample.BuildFromDataFrame(self._scaled_x_data)

    def build_train_and_validation_sets_by_greedy_support_points(self, test_ratio=0.2):

        # Perform Kernel Herding based on the Greedy Support Points algorithm using otkerneldesign for selecting both train and test designs
        # ---------------------------

        # Perform KH on the whole dataset in order to arange it. Then, select the first 'n-validation_size' points to use them for training.
        validation_size = int(self._x_data.getSize() * test_ratio)
        greedy_sp = otkd.GreedySupportPoints(
            candidate_set=self._x_data
        )  # Energy distance kernel
        greedy_sp_design = greedy_sp.select_design(self._x_data.getSize())
        ##greedy_sp_indices = greedy_sp.get_indices(greedy_sp_design)  # code to save in the future
        # WARNING : BUG (Issue #6 here: https://github.com/efekhari27/otkerneldesign/issues)
        greedy_sp_indices = get_indices(
            greedy_sp, greedy_sp_design
        )  # to remove in the future
        print(">> validation_size =", validation_size)

        # Build the learn and validation indices
        learn_indices = greedy_sp_indices[0:-validation_size]
        validation_indices = greedy_sp_indices[validation_size:]

        # Build the learn and validation samples
        self._x_learn = self._x_data[learn_indices]
        self._y_learn = self._y_data[learn_indices]
        self._x_validation = self._x_data[validation_indices]
        self._y_validation = self._y_data[validation_indices]

    def build_and_run_kriging_metamodel(
        self, optimization_algo="LN_COBYLA", nsample_multistart=10
    ):

        # # Compute min-max bounds of the dataset to help for the hyperparameters' optimization
        # # ---------------------------
        # ## Hypothesis: we suppose that the x_data have been already scaled prior to performing ICSCREAM
        # min_bounds = []
        # max_bounds = []
        # weight_bound = 5.0

        # for colname in self._X_Explanatory:
        #     selected_column = self._x_learn.getMarginal([colname])
        #     pairwise_distances = pdist(
        #         np.array(selected_column).reshape(-1, 1), metric="euclidean"
        #     )
        #     min_bounds.append(np.min(pairwise_distances) / weight_bound)
        #     max_bounds.append(np.max(pairwise_distances) * weight_bound)

        # optim_bounds = ot.Interval(min_bounds, max_bounds)
        # ### Probleme : le noyau IsotropicCovarianceModel n'a qu'un seul hyper-parametre
        # ## To do : ajouter une boucle sur les self.X_Secondary_Influential_Inputs_after_aggregation
        # # Idée : Set 10^-6 en valeur min et 50 en valeur max

        # Set the kriging algorithm
        # ---------------------------
        kriging_algo = ot.KrigingAlgorithm(
            self._x_learn,
            self._y_learn,
            self._cov_kriging_model,
            self._kriging_trend_basis,
        )

        # Perform multistart
        # ---------------------------
        ## Hypothesis: we suppose that the x_data have been already scaled prior to performing ICSCREAM
        dist_multi = ot.DistributionCollection()
        for k in range(self._cov_kriging_model.getParameter().getSize() - 1):
            # The "-1" term corresponds to the fact that the amplitude is analytically optimized"
            # See: https://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.GeneralLinearModelAlgorithm.html

            dist_multi.add(
                ot.Uniform(0.0, 1.0)
                # ot.Uniform(
                #     optim_bounds.getLowerBound()[k], optim_bounds.getUpperBound()[k]
                # )
            )

        bounded_dist_multi = ot.JointDistribution(dist_multi)
        one_start = ot.NLopt(optimization_algo)
        one_start.setMaximumEvaluationNumber(10000)

        kriging_algo.setOptimizationAlgorithm(
            ot.MultiStart(
                one_start,
                ## ot.LHSExperiment(bounded_dist_multi, nsample_multistart).generate(),
                ot.LowDiscrepancyExperiment(
                    ot.SobolSequence(), bounded_dist_multi, nsample_multistart, True
                ).generate(),
            )
        )
        # kriging_algo.setOptimizationBounds(optim_bounds) ## Hypothesis: data already scaled

        # Other algorithms:
        # LN_COBYLA | LD_LBFGS | LD_SLSQP
        # See http://openturns.github.io/openturns/master/user_manual/_generated/openturns.NLopt.html#openturns.NLopt
        # Or: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/

        # Run the algorithm
        # ---------------------------
        t = tm.time()
        kriging_algo.run()
        elapsed = tm.time() - t
        print(
            ">> Info: Elapsed time for metamodel training:",
            "{:.6}".format(elapsed),
            "(sec)",
        )

        # Get kriging results and kriging metamodel
        # ---------------------------
        self._kriging_result = kriging_algo.getResult()
        self._kriging_metamodel = self._kriging_result.getMetaModel()

        return kriging_algo

    def validate_kriging_metamodel_using_hold_out_sample(self):

        # Get optimized hyperparameters for trend and covariance
        # ---------------------------
        result_trend = self._kriging_result.getTrendCoefficients()
        result_covariance_model = self._kriging_result.getCovarianceModel()

        kriging_hyperparameters = {
            "trend": result_trend,
            "lengthscale": result_covariance_model.getScale(),  # theta
            "amplitude": result_covariance_model.getAmplitude(),  # sigma_2
            "nugget": result_covariance_model.getNuggetFactor(),
        }

        # Validate the kriging metamodel using the 'ot.MetaModelValidation' class and the hold out validation sample
        # ---------------------------
        self._validation_results = ot.MetaModelValidation(
            self._x_validation, self._y_validation, self._kriging_metamodel
        )

        kriging_residuals = np.array(
            self._validation_results.getResidualSample()
        ).flatten()

        kriging_conditional_variance = np.array(
            self._kriging_result.getConditionalMarginalVariance(self._x_validation)
        )

        # Be careful about the definition of PVA (with/without absolute value)
        signed_predictive_variance_adequacy = np.log10(
            np.sum((kriging_residuals ** 2) / kriging_conditional_variance)
            / len(kriging_residuals)
        )

        validation_metrics = {
            "Q2": self._validation_results.computePredictivityFactor()[0],
            "signed_PVA": signed_predictive_variance_adequacy,
        }

        # Histogram of residuals
        # --------------
        ### WARNING: the followings should be uncommented as soon as the Issue #2655 is solved.
        try:
            ot.HistogramFactory().build(
                self._validation_results.getResidualSample()
            ).drawPDF()
        except RuntimeError:
            print("ON PASSE ICI")
            # Force the bin number in order to avoid explosion.
            ot.HistogramFactory().buildAsHistogram(
                self._validation_results.getResidualSample(), 10
            ).drawPDF()
        graph_histogram_residuals = (
            self._validation_results.getResidualDistribution().drawPDF()
        )
        graph_histogram_residuals.setXTitle("Residuals")
        graph_histogram_residuals.setLegends([""])

        # Observed vs. predicted values
        # --------------
        graph_obs_vs_pred = self._validation_results.drawValidation().getGraph(0, 0)
        graph_obs_vs_pred.setTitle(
            "Obs. vs. Predict -- ($n_{valid}$ = %d)" % self._y_validation.getSize()
        )
        graph_obs_vs_pred.setXTitle("Observed values")
        graph_obs_vs_pred.setYTitle("Predicted values")

        # QQ-plot
        # --------------
        y_predicted_on_validation_sample = self._kriging_metamodel(self._x_validation)
        graph_QQ_plot = ot.VisualTest.DrawQQplot(
            self._y_validation, y_predicted_on_validation_sample
        )
        graph_QQ_plot.setYTitle("Predictions")
        graph_QQ_plot.setTitle("Two sample QQ-plot")

        validation_graphs = {
            "residuals": graph_histogram_residuals,
            "observed_vs_predicted": graph_obs_vs_pred,
            "QQplot": graph_QQ_plot,
        }

        return kriging_hyperparameters, validation_metrics, validation_graphs

    def train_and_validate_kriging_metamodel_using_cross_validation(self):
        """
        TODO
        """
        # def compute_Q2_predictivity_coefficient_by_kfold(self, n_folds=5):
        #     training_sample_size = self.input_kriging_sample.getSize()
        #     splitter = ot.KFoldSplitter(training_sample_size, n_folds)
        #     Q2_score_list = ot.Sample(0, 1)
        #     for indices1, indices2 in splitter:
        #         X_train, X_test = self.input_kriging_sample[indices1], self.input_kriging_sample[indices2]
        #         Y_train, Y_test = self.output_kriging_sample[indices1], self.output_kriging_sample[indices2]
        #         ktrend = ot.ConstantBasisFactory(len(X_train)).build()
        #         marginal_cov_model = ot.MaternModel([1.0], [1.0], 5.0 / 2.0)
        #         cov_X_Exp = ot.ProductCovarianceModel([marginal_cov_model]*len(self.X_Explanatory))
        #         if len(self.X_Secondary_Influential_Inputs) == 0:
        #             kcov = cov_X_Exp
        #         else:
        #             cov_X_SII = ot.IsotropicCovarianceModel(marginal_cov_model, len(self.X_Secondary_Influential_Inputs))
        #             kcov = ot.ProductCovarianceModel([cov_X_Exp, cov_X_SII])
        #         kalgo = ot.KrigingAlgorithm(X_train, Y_train, kcov, ktrend)
        #         kalgo.run()
        #         kresult = kalgo.getResult()
        #         kmm = kresult.getMetaModel()
        #         val = ot.MetaModelValidation(X_test, Y_test, kmm)
        #         Q2_local = val.computePredictivityFactor()[0]
        #         Q2_score_list.add([Q2_local])
        #     Q2_score = Q2_score_list.computeMean()[0]
        #     return Q2_score

    def construct_and_sample_x_tilda_distribution(
        self,
        n_sample_X_Tilda=1000,
        method_compute_m_ebc="AMISE",
    ):
        if self._dist_aleatory is not None:
            # Case #1 - The distribution of X_Tilda is already known
            # --------------
            sample_X_aleatory = self._dist_aleatory.getSample(n_sample_X_Tilda)
            self._sample_X_Tilda = sample_X_aleatory.getMarginal(self._X_Tilda)
        else:
            # Case #2 - The distribution of X_Tilda is not known and should be learnt
            # --------------
            learning_sample_X_Tilda = self._sample_aleatory.getMarginal(self._X_Tilda)
            joint_distribution_X_Tilda = self.fit_distribution_from_sample(
                learning_sample_X_Tilda, method_compute_m_ebc
            )
            self._sample_X_Tilda = joint_distribution_X_Tilda.getSample(
                n_sample_X_Tilda
            )

        return self._sample_X_Tilda

    def construct_and_sample_x_penalized_distribution(
        self,
        n_sample_X_penalized=1000,
        method_compute_m_ebc="AMISE",
    ):
        if self._dist_penalized is not None:
            # Case #1 - The distribution of X_Penalized is already known
            # --------------
            self._sample_X_penalized = self._dist_penalized.getSample(
                n_sample_X_penalized
            )
        else:
            # Case #2 - The distribution of X_Penalized is not known and should be learnt
            # --------------
            joint_distribution_X_penalized = self.fit_distribution_from_sample(
                self._sample_penalized, method_compute_m_ebc
            )
            self._sample_X_penalized = joint_distribution_X_penalized.getSample(
                n_sample_X_penalized
            )

        return self._sample_X_penalized

    @staticmethod
    def fit_distribution_from_sample(learning_sample, method_compute_m_ebc="AMISE"):

        # Fit of the marginals using KDE
        ks = ot.KernelSmoothing()
        ks.setBoundaryCorrection(True)
        list_marginals = []
        for varname in learning_sample.getDescription():
            marg = ks.buildAsTruncatedDistribution(
                learning_sample.getMarginal([varname])
            )
            list_marginals.append(marg)

        # Fit of the copula using the Empirical Bernstein Copula
        ## Methods for the m parameter calculation: "AMISE", "LogLikelihood", "PenalizedCsiszarDivergence"
        bcf = ot.BernsteinCopulaFactory()
        fitted_ebc = bcf.buildAsEmpiricalBernsteinCopula(
            learning_sample, method_compute_m_ebc
        )

        # Create the joint distribution of X_Tilda
        joint_distribution = ot.JointDistribution(list_marginals, fitted_ebc)

        return joint_distribution

    def compute_mean(self):
        return self._kriging_metamodel(self._full_sample).computeMean()[0]

    def compute_1D_conditional_mean(self, varindex, value):

        # Create a new full_sample with a frozen column
        # --------------
        ## WARNING: here, it is supposed that this method uses the index of the considered conditioning variable instead of its name.
        ## WARNING: this method assumes that the frozen variable is mutually independent from the others. This might be a limitation if the variables are dependent.
        full_sample_frozen_column = ot.Sample(self._full_sample)
        full_sample_frozen_column[:, varindex] = [
            [value]
        ] * full_sample_frozen_column.getSize()

        # Apply the predictor to compute 1D conditional mean
        # --------------
        output_sample_frozen_column = self._kriging_metamodel(full_sample_frozen_column)
        one_dimensional_conditional_mean = output_sample_frozen_column.computeMean()

        return one_dimensional_conditional_mean

    def build_1D_conditional_mean(self, varname):

        # Compute varindex from varname
        # --------------
        varindex = self._full_sample_variable_names.index(varname)

        # Create the one_dimensional_conditional_mean_function
        # --------------
        def one_dimensional_conditional_mean_function(x):
            ## WARNING: here, x should be a list of size 1 (useful to create an OpenTURNS PythonFunction)
            return self.compute_1D_conditional_mean(varindex, x[0])

        return one_dimensional_conditional_mean_function

    def build_1D_conditional_mean_as_PythonFunction(self, varname):

        # Goal: create a ot.PythonFunction from a basic function
        # --------------
        basic_function = self.build_1D_conditional_mean(varname)
        pythonfunction = ot.PythonFunction(1, 1, basic_function)
        pythonfunction.setInputDescription([varname])
        pythonfunction.setOutputDescription(["Output Conditional Mean"])

        return pythonfunction

    def compute_2D_conditional_mean(self, varindex1, varindex2, value1, value2):

        # Create a new full_sample with two frozen columns
        # --------------
        ## WARNING: here, it is supposed that this method uses the index of the considered conditioning variable instead of its name.
        ## WARNING: this method assumes that the frozen variables are mutually independent from the others. This might be a limitation if the variables are dependent.
        full_sample_frozen_column = ot.Sample(self._full_sample)
        full_sample_frozen_column[:, varindex1] = [
            [value1]
        ] * full_sample_frozen_column.getSize()
        full_sample_frozen_column[:, varindex2] = [
            [value2]
        ] * full_sample_frozen_column.getSize()

        # Apply the predictor to compute 2D conditional mean
        # --------------
        output_sample_frozen_column = self._kriging_metamodel(full_sample_frozen_column)
        two_dimensional_conditional_mean = output_sample_frozen_column.computeMean()

        return two_dimensional_conditional_mean

    def build_2D_conditional_mean(self, varname1, varname2):

        # Check
        # --------------
        if varname1 == varname2:
            raise ValueError("Arguments 'varname1' and 'varname2' must be different.")

        # Compute varindex from varname
        # --------------
        varindex1 = self._full_sample_variable_names.index(varname1)
        varindex2 = self._full_sample_variable_names.index(varname2)

        # Create the two_dimensional_conditional_mean_function
        # --------------
        def two_dimensional_conditional_mean_function(x):
            ## WARNING: here, x should be a list of size 2 (useful to create an OpenTURNS PythonFunction)
            return self.compute_2D_conditional_mean(varindex1, varindex2, x[0], x[1])

        return two_dimensional_conditional_mean_function

    def build_2D_conditional_mean_as_PythonFunction(self, varname1, varname2):

        # Goal: create a ot.PythonFunction from a basic function
        # --------------
        basic_function = self.build_2D_conditional_mean(varname1, varname2)
        pythonfunction = ot.PythonFunction(2, 1, basic_function)
        pythonfunction.setInputDescription([varname1, varname2])
        pythonfunction.setOutputDescription(["Output Conditional Mean"])

        return pythonfunction

    def compute_allpenalized_conditional_mean(self, values):

        # Create a new full_sample with all frozen columns corresponding to the whole penalized input vector
        # --------------
        full_sample_frozen_column = ot.Sample(self._full_sample)
        penalized_sample_columns = ot.Sample(
            full_sample_frozen_column.getSize(), values
        )
        full_sample_frozen_column[
            :, self._X_Penalized_indices_within_full_sample
        ] = penalized_sample_columns

        # Apply the predictor to compute the GP conditional mean wrt all the penalized inputs
        # --------------
        output_sample_frozen_column = self._kriging_metamodel(full_sample_frozen_column)
        all_penalized_conditional_mean = output_sample_frozen_column.computeMean()

        return all_penalized_conditional_mean

    def build_allpenalized_conditional_mean(self):
        # This function is useful in order to create an OpenTURNS PythonFunction with associated methods.

        # Create the all_penalized_conditional_mean_function
        # --------------
        def all_penalized_conditional_mean_function(x):
            return self.compute_allpenalized_conditional_mean(x)

        return all_penalized_conditional_mean_function

    def build_allpenalized_conditional_mean_as_PythonFunction(self):

        # Goal: create a ot.PythonFunction from a basic function
        # --------------
        basic_function = self.build_allpenalized_conditional_mean()
        pythonfunction = ot.PythonFunction(len(self._X_Penalized), 1, basic_function)
        pythonfunction.setInputDescription(self._X_Penalized)
        pythonfunction.setOutputDescription(["Output Conditional Mean"])

        return pythonfunction

    def compute_conditional_exceedance_probability_from_metamodel(self, full_sample):

        # Apply the metamodel predictor and variance operators
        # --------------
        mean_Gp = self._kriging_result.getConditionalMean(full_sample)
        var_Gp = self._kriging_result.getConditionalMarginalVariance(full_sample)

        # Compute the ratio and integrand
        # --------------
        ratio = (self._empirical_quantile - np.array(mean_Gp)) / np.sqrt(
            np.array(var_Gp)
        )
        integrand = self._standard_normal_distribution.computeSurvivalFunction(ratio)

        # Compute the exceedance probability
        # --------------
        exceedance_probability = integrand.computeMean()

        return exceedance_probability

    def compute_exceedance_probability(self):
        return self.compute_conditional_exceedance_probability_from_metamodel(
            self._full_sample
        )[0]

    def compute_1D_conditional_exceedance_probability(self, varindex, value):

        # Create a new full_sample with a frozen column
        # --------------
        ## WARNING: here, it is supposed that this method uses the index of the considered conditioning variable instead of its name.
        ## WARNING: this method assumes that the frozen variable is mutually independent from the others. This might be a limitation if the variables are dependent.
        full_sample_frozen_column = ot.Sample(self._full_sample)
        full_sample_frozen_column[:, varindex] = [
            [value]
        ] * full_sample_frozen_column.getSize()

        return self.compute_conditional_exceedance_probability_from_metamodel(
            full_sample_frozen_column
        )

    def build_1D_conditional_exceedance_probability(self, varname):

        # Compute varindex from varname
        # --------------
        varindex = self._full_sample_variable_names.index(varname)

        # Create the 1D conditional exceedance probability function
        # --------------
        def one_dimensional_conditional_exceedance_probability(x):
            ## WARNING: here, x should be a list of size 1 (useful to create an OpenTURNS PythonFunction)
            return self.compute_1D_conditional_exceedance_probability(varindex, x[0])

        return one_dimensional_conditional_exceedance_probability

    def build_1D_conditional_exceedance_probability_as_PythonFunction(self, varname):

        # Goal: create a ot.PythonFunction from a basic function
        # --------------
        basic_function = self.build_1D_conditional_exceedance_probability(varname)
        pythonfunction = ot.PythonFunction(1, 1, basic_function)
        pythonfunction.setInputDescription([varname])
        pythonfunction.setOutputDescription(["Conditional Exceedance Probability"])

        return pythonfunction

    def compute_2D_conditional_exceedance_probability(
        self, varindex1, varindex2, value1, value2
    ):

        # Create a new full_sample with two frozen columns
        # --------------
        ## WARNING: here, it is supposed that this method uses the index of the considered conditioning variable instead of its name.
        ## WARNING: this method assumes that the frozen variables are mutually independent from the others. This might be a limitation if the variables are dependent.
        full_sample_frozen_column = ot.Sample(self._full_sample)
        full_sample_frozen_column[:, varindex1] = [
            [value1]
        ] * full_sample_frozen_column.getSize()
        full_sample_frozen_column[:, varindex2] = [
            [value2]
        ] * full_sample_frozen_column.getSize()

        return self.compute_conditional_exceedance_probability_from_metamodel(
            full_sample_frozen_column
        )

    def build_2D_conditional_exceedance_probability(self, varname1, varname2):

        # Check
        # --------------
        if varname1 == varname2:
            raise ValueError("Arguments 'varname1' and 'varname2' must be different.")

        # Compute varindex from varname
        # --------------
        varindex1 = self._full_sample_variable_names.index(varname1)
        varindex2 = self._full_sample_variable_names.index(varname2)

        # Create the two_dimensional_conditional_exceedance_probability_function
        # --------------
        def two_dimensional_conditional_exceedance_probability_function(x):
            ## WARNING: here, x should be a list of size 2 (useful to create an OpenTURNS PythonFunction)
            return self.compute_2D_conditional_exceedance_probability(
                varindex1, varindex2, x[0], x[1]
            )

        return two_dimensional_conditional_exceedance_probability_function

    def build_2D_conditional_exceedance_probability_as_PythonFunction(
        self, varname1, varname2
    ):

        # Goal: create a ot.PythonFunction from a basic function
        # --------------
        basic_function = self.build_2D_conditional_exceedance_probability(
            varname1, varname2
        )
        pythonfunction = ot.PythonFunction(2, 1, basic_function)
        pythonfunction.setInputDescription([varname1, varname2])
        pythonfunction.setOutputDescription(["Conditional Exceedance Probability"])

        return pythonfunction

    def compute_allpenalized_conditional_exceedance_probability(self, values):

        # Create a new full_sample with all frozen columns corresponding to the whole penalized input vector
        # --------------
        full_sample_frozen_column = ot.Sample(self._full_sample)
        penalized_sample_columns = ot.Sample(
            full_sample_frozen_column.getSize(), values
        )
        full_sample_frozen_column[
            :, self._X_Penalized_indices_within_full_sample
        ] = penalized_sample_columns

        return self.compute_conditional_exceedance_probability_from_metamodel(
            full_sample_frozen_column
        )

    def build_allpenalized_conditional_exceedance_probability(self):
        # This function is useful in order to create an OpenTURNS PythonFunction with associated methods.

        # Create the allpenalized_conditional_exceedance_probability_function
        # --------------
        def allpenalized_conditional_exceedance_probability_function(x):
            return self.compute_allpenalized_conditional_exceedance_probability(x)

        return allpenalized_conditional_exceedance_probability_function

    def build_allpenalized_conditional_exceedance_probability_as_PythonFunction(self):

        # Goal: create a ot.PythonFunction from a basic function
        # --------------
        basic_function = self.build_allpenalized_conditional_exceedance_probability()
        pythonfunction = ot.PythonFunction(len(self._X_Penalized), 1, basic_function)
        pythonfunction.setInputDescription(self._X_Penalized)
        pythonfunction.setOutputDescription(["Conditional Exceedance Probability"])

        return pythonfunction

    def create_full_sample_for_metamodel_prediction(self):

        ## WARNING: the term 'full_sample' denotes all the inputs involved in the GP metamodel regression
        ## Create a full sample for the metamodel to be used to compute several quantities
        full_sample = ot.Sample(
            np.hstack([self._sample_X_penalized, self._sample_X_Tilda])
        )
        full_sample.setDescription(self._X_Penalized + self._X_Tilda)

        # Modify the order of the inputs wrt the ranking order obtained from GSA/TSA
        self._full_sample_variable_names = (
            self._X_Explanatory + self._X_Secondary_Influential_Inputs_after_aggregation
        )
        self._full_sample = full_sample.getMarginal(self._full_sample_variable_names)

        # Get the positions of the X_penalized inputs within the full_sample
        self._X_Penalized_indices_within_full_sample = [
            ind
            for ind, item in enumerate(self._X_Explanatory)
            if item in self._X_Penalized
        ]

    ## REMARK: we still enable the user to define proper OpenTURNS' PythonFunction objects or to call directly the Python functions already defined to use them
    ## exampleuse = ot.PythonFunction(1,1,icscream.build_1D_conditional_mean(x13))
