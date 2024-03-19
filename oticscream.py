#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2024

@author: Vincent Chabridon
"""
import os

work_path = "/home/g44079/Documents/01_PROJETS/THCOEURS/oticscream/oticscream"
os.chdir(work_path)
import openturns as ot

print("OpenTURNS version:", ot.__version__)
import openturns.viewer as otv
import numpy as np
from scipy.spatial.distance import pdist
import pandas as pd
import matplotlib.pyplot as plt

#plt.rcParams["text.usetex"] = True
# from matplotlib import rc, rcParams, stylercParams['text.usetex'] = Truerc('font', **{'family': 'Times'})rc('text', usetex=True)rc('font', size=16)# Set the default text font sizerc('axes', titlesize=20)# Set the axes title font sizerc('axes', labelsize=16)# Set the axes labels font sizerc('xtick', labelsize=14)# Set the font size for x tick labelsrc('ytick', labelsize=16)# Set the font size for y tick labelsrc('legend', fontsize=16)# Set the legend font size`
import time as tm

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


class Icscream:
    """
    Description TODO.

    Parameters
    ----------
    random_distribution : :class:`openturns.Distribution`
    scenario_distribution : :class:`openturns.Distribution`
    model : function
    dataset : pd.DataFrame
    random_variables_columns : list
    scenario_variables_columns : list
    output_variable_column : list
    output_quantile_order : scalar
    p_value_threshold : scalar
    n_sim : integer

    Examples
    --------
    >>> todo
    """

    def __init__(
        self,
        df_scenario=None,
        df_aleatory=None,
        df_output=None,
        covariance_collection=None,
        output_quantile_order=0.9,
        p_value_threshold=0.05,
        n_perm=200,
    ):
        if (
            (df_scenario is None)
            or (df_aleatory is None)
            or (df_output is None)
        ):
            raise ValueError(
                "Please, provide lists corresponding to 'df_scenario'', 'df_aleatory' and 'df_output'."
            )
        else:
            self._df_scenario = df_scenario
            self._df_aleatory = df_aleatory

            self._df_input = pd.concat([self._df_scenario, self._df_aleatory], axis=1)
            self._df_output = df_output

            self._sample_scenario = ot.Sample(
                self._df_scenario.values
            )
            self._sample_aleatory = ot.Sample(
                self._df_aleatory.values
            )
            self._sample_output = ot.Sample(
                self._df_output.values
            )
            self._sample_input = ot.Sample(
                pd.concat([self._sample_scenario, self._sample_aleatory], axis=1)
            )
            self._dim_scenario = self._sample_scenario.getDimension()
            self._dim_random = self._sample_aleatory.getDimension()
            self._dim_input = self._sample_input.getDimension()

        self._output_quantile_order = output_quantile_order
        self._empirical_quantile = self._sample_output.computeQuantile(
            self._output_quantile_order
        )[0]
        self._p_value_threshold = p_value_threshold
        self._n_perm = n_perm

        if covariance_collection is None:
            input_covariance_collection = []
            for i in range(self._dim_input):
                Xi = self.input_sample.getMarginal(i)
                input_covariance = ot.SquaredExponential(1)
                input_covariance.setScale(Xi.computeStandardDeviation())
                input_covariance_collection.append(input_covariance)
            output_covariance = ot.SquaredExponential(1)
            output_covariance.setScale(
                self._sample_output.computeStandardDeviation()
            )
            self._covariance_collection = input_covariance_collection + [output_covariance]
        else:
            self._covariance_collection = covariance_collection            

        self._GSA_study = None
        self._TSA_study = None
        self._CSA_study = None

        df_columns = self._df_scenario.columns.tolist() + self._df_aleatory.columns.tolist()

        self._GSA_results = pd.DataFrame([],columns=df_columns,index=[],)
        self._TSA_results = pd.DataFrame([],columns=df_columns,index=[],)
        self._CSA_results = pd.DataFrame([],columns=df_columns,index=[],)
        self._Aggregated_pval_results = pd.DataFrame([],columns=df_columns,index=[],)

        self._X_Primary_Influential_Inputs = None
        self._X_Secondary_Influential_Inputs = None
        self._X_Secondary_Influential_Inputs_after_aggregation = None
        self._X_Epsilon = None
        self._X_Penalized = None
        self._X_Explanatory = None
        self._X_Tilda = None
        self._x_data = None
        self._y_data = None
        self._scaled_x_data = None
        self._input_kriging_sample = None
        self._output_kriging_sample = None
        self._kriging_result = None
        self._kriging_metamodel = None
        self._validation_results = None
        self._X_Penalized_data = None
        self._X_Penalized_sample = None
        # self.Conditional_Probabilities_Results = pd.DataFrame([], columns = self.X_Penalized, index=[])
        self._list_probabilities_x_pen = None

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
        median_line = ot.Curve([[median_Y], [median_Y]], [[0.0], [fit.computePDF([median_Y])]])
        median_line.setLineWidth(2.0)
        graph_output.add(median_line)
        #
        quantile_line = ot.Curve([[self._empirical_quantile], [self._empirical_quantile]], [[0.0], [fit.computePDF([self._empirical_quantile])]])
        quantile_line.setLineWidth(2.0)
        graph_output.add(quantile_line)
        #
        #graph_output.setColors(["dodgerblue3", "darkorange1"])
        graph_output.setLegends(["Histogram", "KDE", "Mean", "Median", "Empirical quantile"])
        graph_output.setLegendPosition("topright")
        graph_output.setTitle("Empirical quantile =", "{:.6}".format(self._empirical_quantile))
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
            self.covariance_collection,
            self.input_sample,
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
            self.covariance_collection,
            self.input_sample,
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

        graph_dict = {"GSA indices" :    graph_GSA_indices,
                      "GSA pval asymp" : graph_GSA_pval_asymp,
                      "GSA pval perm" :  graph_GSA_pval_perm,
                      "TSA indices" :    graph_TSA_indices,
                      "TSA pval asymp" : graph_TSA_pval_asymp,
                      "TSA pval perm" :  graph_TSA_pval_perm,
                      "CSA indices" :    graph_CSA_indices,
                      "CSA pval perm" :  graph_CSA_pval_perm,
                      }
        
        return graph_dict

    def aggregate_pvalues_and_sort_variables(self, isAsymptotic=False,
                                             sortby_method="Bonferroni"):
        if self._GSA_study is None:
            _ = self.perform_GSA_study()
        if self._TSA_study is None:
            _ = self.perform_TSA_study()
        if self._CSA_study is None:
            _ = self.perform_CSA_study()

        sortby_method += " p-values"

        if isAsymptotic:
            print(">> Info: Asymptotic regime => asymptotic p-values will be used!")
            self._Aggregated_pval_results.loc["GSA p-values", :] = self._GSA_results.loc[
                "p-values asymp", :
            ]
            self._Aggregated_pval_results.loc["TSA p-values", :] = self._TSA_results.loc[
                "p-values asymp", :
            ]
        else:
            print(
                ">> Info: Non-asymptotic regime => permutation-based p-values will be used!"
            )
            self._Aggregated_pval_results.loc["GSA p-values", :] = self._GSA_results.loc[
                "p-values perm", :
            ]
            self._Aggregated_pval_results.loc["TSA p-values", :] = self._TSA_results.loc[
                "p-values perm", :
            ]
            # TODO : inclure  les p-valeurs CSA dans la stratégie globale
            self._Aggregated_pval_results.loc["CSA p-values", :] = self._CSA_results.loc[
                "p-values perm", :
            ]

        pval_GSA = self._Aggregated_pval_results.loc["GSA p-values", :].values
        pval_TSA = self._Aggregated_pval_results.loc["TSA p-values", :].values

        # Aggregate GSA and TSA results using Bonferroni correction
        # ---------------------------
        self._Aggregated_pval_results.loc["Bonferroni p-values", :] = np.minimum(
            2 * np.minimum(pval_GSA, pval_TSA), 1
        )

        # Other advanced statistics used for aggregation
        # ---------------------------
        self._Aggregated_pval_results.loc["Fisher p-values", :] = -np.log(
            pval_GSA.astype("float64")
        ) - np.log(pval_TSA.astype("float64"))
        self._Aggregated_pval_results.loc["Tippet p-values", :] = 1 - np.minimum(
            pval_GSA, pval_TSA
        )

        for i, colname in enumerate(self._df_input.columns):
            invgamma_dist_GSA = ot.InverseGamma(1 / pval_GSA[i], 1.0)
            invgamma_dist_TSA = ot.InverseGamma(1 / pval_TSA[i], 1.0)
            self._Aggregated_pval_results.loc[
                "InvGamma p-values", colname
            ] = invgamma_dist_GSA.computeCDF(
                1 - pval_GSA[i]
            ) + invgamma_dist_TSA.computeCDF(
                1 - pval_TSA[i]
            )

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

        inputs_dict = {"X_Primary_Influential_Inputs" : self._X_Primary_Influential_Inputs,
                       "X_Secondary_Influential_Inputs" : self._X_Secondary_Influential_Inputs,
                       "X_Epsilon" : self._X_Epsilon,
                       }

        return inputs_dict

    def build_and_validate_kriging_metamodel(
        self, nugget_factor=1e-6, optimization_algo="LN_COBYLA", nsample_multistart=10
    ):
        self._X_Penalized = self.scenario_variables_columns
        print(">> X_Penalized =", self._X_Penalized)

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

        print(">> X_Explanatory =", self._X_Explanatory)
        print(
            ">> X_Secondary_Influential_Inputs after aggregation =",
            self._X_Secondary_Influential_Inputs_after_aggregation,
        )
        print(
            ">> X_Tilda =", self._X_Tilda
        )  # Useful for conditional probability computation

        if len(self._X_Explanatory) > 30:
            raise ValueError("The sequential strategy should be implemented!")
        else:
            print(">> Info: Direct metamodel building strategy will be used!")

        # Loading data
        # ---------------------------
        self._x_data = self.dataset[
            self._X_Explanatory + self._X_Secondary_Influential_Inputs_after_aggregation
        ]
        self._y_data = self.dataset[self.output_variable_column]
        dim_krig = len(self._X_Explanatory) + len(
            self._X_Secondary_Influential_Inputs_after_aggregation
        )
        print(">> dim_krig =", dim_krig)

        # Scaling data
        # ---------------------------
        self._scaled_x_data = (self._x_data - self._x_data.mean(axis=0)) / self._x_data.std(
            axis=0
        )
        scaled_x_data_sample = ot.Sample.BuildFromDataFrame(self._scaled_x_data)

        # Trend
        # ---------------------------
        kriging_trend_basis = ot.ConstantBasisFactory(dim_krig).build()
        # basis = ot.LinearBasisFactory(dim_krig).build()
        # basis = ot.QuadraticBasisFactory(dim_krig).build()

        # Covariance model for the explanatory inputs (X_EXP)
        # ---------------------------
        marginal_cov_model = ot.MaternModel([1.0], [1.0], 5.0 / 2.0)
        # ICSCREAM: X_Explanatory in a tensorized stationary anisotropic covariance
        ## Modif VCN : on ajoute les X_Secondary_Influential_Inputs_after_aggregation dans le ProductCovarianceModel
        # old cov_X_Explanatory = ot.ProductCovarianceModel([marginal_cov_model]*len(self.X_Explanatory))
        cov_X_Explanatory = ot.ProductCovarianceModel([marginal_cov_model] * dim_krig)

        cov_kriging_model = cov_X_Explanatory

        ### Test commenté si jamais il faut distinguer X_EXP et X_SII (+ retirer ligne 395)
        # if len(self.X_Secondary_Influential_Inputs_after_aggregation) == 0:
        #     print(">> Info: X_Secondary_Influential_Inputs is empty and thus will not be considered in the covariance model")
        #     cov_kriging_model = cov_X_Explanatory
        # else:
        #     print(">> Info: X_Secondary_Influential_Inputs is not empty and thus will be modeled through a stationary isotropic covariance model")
        #     # ICSCREAM: X_Secondary_Influential_Inputs joint in a stationary isotropic covariance
        #     # ----------------------------
        #     # WARNING : si  ot.IsotropicCovarianceModel() utilisé, alors il faut
        #     ## TODO : cov_X_Secondary_Influential_Inputs = ot.IsotropicCovarianceModel(marginal_cov_model, len(self.X_Secondary_Influential_Inputs_after_aggregation))
        #     # Resulting covariance model
        #     # cov_kriging_model = ot.ProductCovarianceModel([cov_X_Explanatory, cov_X_Secondary_Influential_Inputs])

        # Set nugget factor
        # ---------------------------
        cov_kriging_model.setNuggetFactor(nugget_factor)

        # [NOT USED FOR NOW] Covariance model for the remaining inputs (X_EPS)
        # ---------------------------
        # ICSCREAM: X_EPS captured via an homoscedastic nugget effect
        # Cannot use a DiracCovarianceModel here, see issue #1857
        # https://github.com/openturns/openturns/issues/1857
        # cov3 = ot.DiracCovarianceModel(n3)
        ###########################
        # [NOT USED FOR NOW] epsTol = 1e-8
        # [NOT USED FOR NOW] cov_X_EPS = ot.AbsoluteExponential([epsTol]*nb_eps)
        # [NOT USED FOR NOW] cov_X_EPS.setActiveParameter([nb_eps])
        ###########################
        # On active uniqt l'amplitude pour avoir la bonne valeur du nugget
        # (de 0 à nb_eps-1, ce sont des valeurs de theta.
        # A nb_eps, on active la valeur de l'amplitude)

        # Perform Kernel Herding using otkerneldesign for selecting both train and test designs
        # ---------------------------
        test_ratio = 0.2
        test_size = int(self._scaled_x_data.shape[0] * test_ratio)
        kernel_herding = otkd.KernelHerding(
            kernel=cov_kriging_model, candidate_set=scaled_x_data_sample
        )
        kernel_herding_design = kernel_herding.select_design(test_size)
        kernel_herding_indices = kernel_herding.get_indices(kernel_herding_design)
        print(">> test_size =", test_size)

        x_test_kh = kernel_herding_design
        y_test_kh = self._y_data.loc[kernel_herding_indices]
        copy_scaled_x_data = self._scaled_x_data.copy()
        copy_y_data = self._y_data.copy()
        x_learn_kh = copy_scaled_x_data.drop(
            kernel_herding_indices, axis=0, inplace=False
        )
        y_learn_kh = copy_y_data.drop(kernel_herding_indices, axis=0, inplace=False)
        print(
            ">> Check if 'y_test_kh' and 'y_learn_kh' are disjoint (0 if True) =",
            np.sum(np.isin(y_test_kh, y_learn_kh)),
        )

        # Compute min and max bounds of the dataset
        # ---------------------------
        min_bounds = []
        max_bounds = []
        weight_bound = 5.0

        for colname in self._scaled_x_data.columns:
            selected_column = self._scaled_x_data[colname]
            pairwise_distances = pdist(
                np.array(selected_column).reshape(-1, 1), metric="euclidean"
            )
            min_bounds.append(np.min(pairwise_distances) / weight_bound)
            max_bounds.append(np.max(pairwise_distances) * weight_bound)

        optim_bounds = ot.Interval(min_bounds, max_bounds)
        ### Probleme : le noyau IsotropicCovarianceModel n'a qu'un seul hyper-parametre
        ## To do : ajouter une boucle sur les self.X_Secondary_Influential_Inputs_after_aggregation
        # Idée : Set 10^-6 en valeur min et 50 en valeur max

        # Set the kriging algorithm
        # ---------------------------
        self._input_kriging_sample = ot.Sample.BuildFromDataFrame(x_learn_kh)
        self._output_kriging_sample = ot.Sample.BuildFromDataFrame(y_learn_kh)
        kriging_algo = ot.KrigingAlgorithm(
            self._input_kriging_sample,
            self._output_kriging_sample,
            cov_kriging_model,
            kriging_trend_basis,
        )

        # Perform multistart
        # ---------------------------
        dist_multi = ot.DistributionCollection()
        for k in range(dim_krig):
            dist_multi.add(
                ot.Uniform(
                    optim_bounds.getLowerBound()[k], optim_bounds.getUpperBound()[k]
                )
            )

        bounded_dist_multi = ot.ComposedDistribution(dist_multi)

        kriging_algo.setOptimizationAlgorithm(
            ot.MultiStart(
                ot.NLopt(optimization_algo),
                ot.LHSExperiment(bounded_dist_multi, nsample_multistart).generate(),
            )
        )
        kriging_algo.setOptimizationBounds(optim_bounds)

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

        result_trend = self._kriging_result.getTrendCoefficients()
        result_covariance_model = self._kriging_result.getCovarianceModel()

        covariance_length_scale = np.array(result_covariance_model.getScale())
        covariance_amplitude = np.array(result_covariance_model.getAmplitude())

        print(">> Result trend =", result_trend[0])
        print(">> Length scale (theta) =", covariance_length_scale)
        print(">> Amplitude (sigma^2) =", covariance_amplitude)

        # Validate the kriging metamodel using the 'ot.MetaModelValidation' class
        # ---------------------------
        y_test_kh_sample = ot.Sample.BuildFromDataFrame(y_test_kh)
        self._validation_results = ot.MetaModelValidation(
            x_test_kh, y_test_kh_sample, self._kriging_metamodel
        )

        Q2_coefficient = self._validation_results.computePredictivityFactor()[0]
        print(">> Q2 =", "{:.6}".format(Q2_coefficient))

        kriging_residuals = np.array(
            self._validation_results.getResidualSample()
        ).flatten()
        kriging_conditional_variance = np.array(
            self._kriging_result.getConditionalMarginalVariance(x_test_kh)
        )
        Predictive_Variance_Adequacy = np.abs(
            np.log10(
                np.sum((kriging_residuals**2) / kriging_conditional_variance)
                / len(kriging_residuals)
            )
        )
        print(">> PVA =", "{:.6}".format(Predictive_Variance_Adequacy))

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

        # Histogram of residuals
        # --------------
        residuals = self._validation_results.getResidualSample()
        ot.HistogramFactory().build(residuals).drawPDF()
        graph_histogram_residuals = (
            self._validation_results.getResidualDistribution().drawPDF()
        )
        graph_histogram_residuals.setXTitle("Residuals")
        graph_histogram_residuals.setLegends("")
        view_residuals = otv.View(graph_histogram_residuals)
        view_residuals.save(
            self.figpath + "kriging_residuals_histogram.png",
            dpi=150,
            bbox_inches="tight",
        )
        view_residuals.save(
            self.figpath + "kriging_residuals_histogram.pdf",
            dpi=150,
            bbox_inches="tight",
        )

        # Observed vs. predicted values
        # --------------
        graph_observed_vs_predicted = self._validation_results.drawValidation()
        graph_observed_vs_predicted.setTitle(
            "Obs. vs. Predict -- ($n_{valid}$ = %d)" % test_size
        )
        view_observed_vs_predicted = otv.View(graph_observed_vs_predicted)
        view_observed_vs_predicted.save(
            self.figpath + "kriging_observed_vs_predicted.png",
            dpi=150,
            bbox_inches="tight",
        )
        view_observed_vs_predicted.save(
            self.figpath + "kriging_observed_vs_predicted.pdf",
            dpi=150,
            bbox_inches="tight",
        )

        # QQ-plot
        # --------------
        y_predicted_on_test_sample = self._kriging_metamodel(x_test_kh)
        graph_QQ_plot = ot.VisualTest.DrawQQplot(
            y_test_kh_sample, y_predicted_on_test_sample
        )
        graph_QQ_plot.setXTitle("Validation data")
        graph_QQ_plot.setYTitle("Predictions")
        graph_QQ_plot.setTitle("Two sample QQ-plot")
        view_QQ_plot = otv.View(graph_QQ_plot)
        view_QQ_plot.save(
            self.figpath + "kriging_QQ_plot.png", dpi=150, bbox_inches="tight"
        )
        view_QQ_plot.save(
            self.figpath + "kriging_QQ_plot.pdf", dpi=150, bbox_inches="tight"
        )

    def compute_conditional_probabilities(
        self, composed_distribution_X_Tilda, n_sample_X_Tilda=100
    ):
        # Sample X_Tilda according to its distribution
        # --------------
        sample_X_Tilda = composed_distribution_X_Tilda.getSample(n_sample_X_Tilda)
        sample_X_Tilda.setDescription(self._X_Tilda)

        # Discretize intervals of penalized inputs
        # --------------
        self._X_Penalized_data = self.dataset[self._X_Penalized]
        self._X_Penalized_sample = ot.Sample.BuildFromDataFrame(self._X_Penalized_data)

        min_penalized = []
        max_penalized = []

        for colname in self._X_Penalized_data.columns:
            selected_column = self._X_Penalized_data[colname]
            array_column = np.array(selected_column).reshape(-1, 1)
            min_penalized.append(np.min(array_column))
            max_penalized.append(np.max(array_column))

        bounds_penalized = ot.Interval(min_penalized, max_penalized)

        dist_penalized = ot.DistributionCollection()
        for k in range(len(self._X_Penalized)):
            dist_penalized.add(
                ot.Uniform(
                    bounds_penalized.getLowerBound()[k],
                    bounds_penalized.getUpperBound()[k],
                )
            )

        bounded_distribution_penalized = ot.ComposedDistribution(dist_penalized)

        new_sample_X_Penalized = bounded_distribution_penalized.getSample(
            n_sample_X_Tilda
        )
        new_sample_X_Penalized.setDescription(self._X_Penalized)
        # print("new_sample_X_Penalized =", new_sample_X_Penalized)

        array_X_Penalized = np.array(new_sample_X_Penalized)
        # print("array_X_Penalized =", array_X_Penalized)

        # Compute conditional probabilities
        # --------------
        self._list_probabilities_x_pen = []

        for k in range(len(self._X_Penalized)):
            results_probabilities = {}
            print("k =", k)
            array_column = np.array(new_sample_X_Penalized[:, k]).reshape(-1, 1)
            # print(array_column)
            for j in array_column:
                # print("j =", j)
                x_pen = j[0]
                # print("x_pen =", x_pen)
                # x_pen = np.repeat([x_pen], n_sample_X_Tilda, axis=0)
                x_pen_tot = array_X_Penalized.copy()
                x_pen_tot[:, k] = x_pen
                # print("array_X_Penalized =", array_X_Penalized)
                # print("x_pen_tot =", x_pen_tot)
                x_joint = np.concatenate([x_pen_tot, np.array(sample_X_Tilda)], axis=1)
                # print("x_joint =", x_joint)

                mean_Gp = self._kriging_result.getConditionalMean(x_joint)
                var_Gp = self._kriging_result.getConditionalMarginalVariance(x_joint)
                ratio_integrand = (
                    self._empirical_quantile - np.array(mean_Gp)
                ) / np.sqrt(np.array(var_Gp))
                proba_x_pen = 1 - np.mean(ot.Normal().computeCDF(ratio_integrand))

                # self.Conditional_Probabilities_Results.loc[j, k] = proba_x_pen

                results_probabilities["proba_x_pen_" + str(j[0])] = [j[0], proba_x_pen]
                # print("results_probabilities =", self.results_probabilities)
                # list_probabilities.append(proba_x_pen)
            self._list_probabilities_x_pen.append([k, results_probabilities])

        # Draw conditional probabilities
        # --------------
        plt.figure()
        for k in range(len(self._X_Penalized)):
            a = self._list_probabilities_x_pen[k][1].values()
            values_array = np.array([i for i in a])
            b = values_array[values_array[:, 0].argsort()]
            x = b[:, 0]
            y = b[:, 1]
            plt.xlabel("X_{pen} range")
            plt.ylabel("proba")
            plt.scatter(x, y, marker="o")
            plt.plot(x, y, linestyle="--")
            # plt.grid(axis='x', color='0.95')
            plt.legend()
            plt.grid(True)
            # plt.title('le titre')
        plt.show()
        plt.savefig(
            self.figpath + "Conditional_Probabilities.png", dpi=150, bbox_inches="tight"
        )
        plt.savefig(
            self.figpath + "Conditional_Probabilities.pdf", dpi=150, bbox_inches="tight"
        )

        # plt.plot(xi, y, marker='o', linestyle='--', color='r', label='Square')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.xticks(xi, x)
        # plt.title('compare')
        # plt.legend()
        # plt.show()

        # import numpy as np
        # import matplotlib.pyplot as plt

        # x = np.arange(14)
        # y = np.sin(x / 2)

        # plt.step(x, y + 2, label='pre (default)')
        # plt.plot(x, y + 2, 'o--', color='grey', alpha=0.3)

        # plt.step(x, y + 1, where='mid', label='mid')
        # plt.plot(x, y + 1, 'o--', color='grey', alpha=0.3)

        # plt.step(x, y, where='post', label='post')
        # plt.plot(x, y, 'o--', color='grey', alpha=0.3)

        # plt.grid(axis='x', color='0.95')
        # plt.legend(title='Parameter where:')
        # plt.title('plt.step(where=...)')
        # plt.show()

        # ## Attention -> Ajouter un Check sur l'ordre de la concaténation des X_PEN / X_TILDA
        # ## Bonne pratique => toujours mettre les X_PEN en premier au départ d'oticscream
