import os
import seaborn as sns
import numpy as np
import pandas
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from pingouin import multicomp
from scipy.stats import fisher_exact
from scipy.stats.distributions import chi2
from statsmodels.stats.contingency_tables import Table2x2


def likelihood_ratio(ll0, ll1):
    return -2 * (ll0-ll1)

def interaction_p_value(feature_table, clinical_table, image_biomarker, clinical_biomarkers,treatment, outcome_title):
    combined_table = pd.concat([feature_table, clinical_table], axis=1)[
        clinical_biomarkers + [image_biomarker, treatment, outcome_title]].dropna()
    combined_table[image_biomarker + '_interaction'] = combined_table[image_biomarker] * combined_table[treatment]

    variables = [treatment,image_biomarker]+clinical_biomarkers
    model_descriptions = outcome_title + ' ~ ' + ' + '.join(variables)
    model0 = smf.logit(model_descriptions, data=combined_table).fit(disp=0)
    variables.append(image_biomarker + '_interaction')
    model_descriptions = outcome_title + ' ~ ' + ' + '.join(variables)
    model1 = smf.logit(model_descriptions, data=combined_table).fit(disp=0)
    L0, L1 = model0.llf, model1.llf
    df0, df1 = model0.df_model, model1.df_model

    chi2_stat = likelihood_ratio(L0, L1)
    p = chi2.sf(chi2_stat, df1 - df0)
    return p

# def univariate_p_value(feature_table, clinical_table, biomarker, outcome):
#     features = feature_table[biomarker]
#     outcomes = clinical_table[outcome]
#     table = pd.crosstab(features, outcomes)
#     # contigency_table = Table2x2(table)
#     # odds = contigency_table.oddsratio
#     oddsratio, pvalue = fisher_exact(table)
#     return pvalue

def univariate_p_value(feature_table, clinical_table, biomarker, outcome_title):
    combined_table = pd.concat([feature_table, clinical_table], axis=1)[[biomarker, outcome_title]].dropna()
    model_descriptions = outcome_title + ' ~ ' + biomarker
    model = smf.logit(model_descriptions, data=combined_table).fit(disp=0)
    stats = model.wald_test('({0} = 0)'.format(biomarker), scalar=True)
    # print(stats.pvalue)
    return stats.pvalue



def logistic_regression_multivariate(feature_table, clinical_table, treatment, image_biomarkers, clinical_biomarkers,
                                     outcome_title, with_interaction = False):
    if isinstance(image_biomarkers, str):
        image_biomarkers = [image_biomarkers]
    combined_table = pd.concat([feature_table, clinical_table], axis=1)[clinical_biomarkers+image_biomarkers+[treatment,outcome_title]].dropna()
    # for biomarker in biomarkers:
    if with_interaction:
        variables = clinical_biomarkers + image_biomarkers + [treatment]
        for image_biomarker in image_biomarkers:
            variables.append(image_biomarker + '_interaction')
            combined_table[image_biomarker+'_interaction'] = combined_table[image_biomarker]*combined_table[treatment]
    else:
        variables = clinical_biomarkers + image_biomarkers
    # univariate_results = {}
    # for variable in variables:
    #     # decorated_variable = 'Q("' + str(variable) + '")'
    #     model_descriptions = outcome_title + ' ~ ' + variable
    #     model = smf.logit(model_descriptions, data=combined_table).fit()
    #     # print(model.params)
    #     result = pd.Series(
    #         {
    #             "OR": np.exp(model.params[variable]),
    #             "Lower CI": np.exp(model.conf_int()[0][variable]),
    #             "Upper CI": np.exp(model.conf_int()[1][variable]),
    #             'p-value': model.pvalues[variable]
    #         }
    #     )
    #     univariate_results[variable] = result
    # univariate_results = pd.concat(univariate_results, axis=1).T

    model_descriptions = outcome_title+ ' ~ '+ ' + '.join(variables)
    model = smf.logit(model_descriptions, data=combined_table).fit()
    # view model summary
    # print(model.summary())
    odds_ratios = pd.DataFrame(
        {
            "OR": model.params,
            "Lower CI": model.conf_int()[0],
            "Upper CI": model.conf_int()[1],
        }
    )
    odds_ratios = np.exp(odds_ratios)
    pvalues = model.pvalues
    pvalues.name = 'p-value'

    multivariate_results = pd.concat([odds_ratios, pvalues],axis=1)
    # print(multivariate_results)
    # multivariate_results.index = variables
    return multivariate_results
    # combined_results = pd.concat([univariate_results, multivariate_results], axis=1, keys=['Univariate','Multivariate'])
    # return combined_results

def image_biomarker_discovery_pipeline(image_feature_table, clinical_feature_table, treatment, clinical_biomarkers,
                                       outcome_title, export_directory, p_value_correction=True):
    # image_feature_table = (image_feature_table-image_feature_table.mean())/image_feature_table.std()
    p_values = {}
    clinical_feature_table_subset = clinical_feature_table[clinical_feature_table[treatment] == 1]
    for feature_name in image_feature_table.columns.values:
        p_value = univariate_p_value(image_feature_table.loc[clinical_feature_table_subset.index,:],
                                                                clinical_feature_table_subset, feature_name, outcome_title)
        # print(univariate_results)
        # print(multivariate_results)
        p_values[feature_name] = p_value
    p_values = pd.Series(p_values, name='p-value')
    reject, corrected_p_values = multicomp(p_values.values, 0.05, method='fdr_bh')
    corrected_p_values = pd.Series(corrected_p_values, index=p_values.index, name = 'p-value corrected')
    univariate_treatment_stats = pd.concat([p_values, corrected_p_values], axis=1)
    if p_value_correction:
        selected_features = corrected_p_values[corrected_p_values<0.05].index
    else:
        selected_features = p_values[p_values < 0.05].index

    p_values = {}
    clinical_feature_table_subset = clinical_feature_table[clinical_feature_table[treatment] == 0]
    for feature_name in image_feature_table.columns.values:
        p_value = univariate_p_value(image_feature_table.loc[clinical_feature_table_subset.index],
                                     clinical_feature_table_subset, feature_name, outcome_title)
        # print(univariate_results)
        # print(multivariate_results)
        p_values[feature_name] = p_value
    p_values = pd.Series(p_values, name='p-value')
    reject, corrected_p_values = multicomp(p_values.values, 0.05, method='fdr_bh')
    corrected_p_values = pd.Series(corrected_p_values, index=p_values.index, name = 'p-value corrected')
    univariate_control_stats = pd.concat([p_values, corrected_p_values], axis=1)

    p_values = {}
    for feature_name in image_feature_table.columns.values:
        p_value = interaction_p_value(image_feature_table, clinical_feature_table, feature_name,
                                      [],treatment, outcome_title)
        # print(univariate_results)
        # print(multivariate_results)
        p_values[feature_name] = p_value
    p_values = pd.Series(p_values, name='p-value')
    reject, corrected_p_values = multicomp(p_values.values, 0.05, method='fdr_bh')
    corrected_p_values = pd.Series(corrected_p_values, index=p_values.index, name = 'p-value corrected')
    multivariate_interaction_stats = pd.concat([p_values, corrected_p_values], axis=1)
    if p_value_correction:
        selected_features = corrected_p_values[corrected_p_values < 0.05].index.intersection(selected_features)
    else:
        selected_features = p_values[p_values < 0.05].index.intersection(selected_features)

    p_values = {}
    for feature_name in image_feature_table.columns.values:
        p_value = interaction_p_value(image_feature_table, clinical_feature_table, feature_name,
                                      clinical_biomarkers, treatment, outcome_title)
        # print(univariate_results)
        # print(multivariate_results)
        p_values[feature_name] = p_value
    p_values = pd.Series(p_values, name='p-value')
    reject, corrected_p_values = multicomp(p_values.values, 0.05, method='fdr_bh')
    corrected_p_values = pd.Series(corrected_p_values, index=p_values.index, name = 'p-value corrected')
    multivariate_interaction_corrected_stats = pd.concat([p_values, corrected_p_values], axis=1)
    if p_value_correction:
        selected_features = corrected_p_values[corrected_p_values < 0.05].index.intersection(selected_features)
    else:
        selected_features = p_values[p_values < 0.05].index.intersection(selected_features)

    summary = pd.concat([univariate_treatment_stats, univariate_control_stats, multivariate_interaction_stats, multivariate_interaction_corrected_stats],
                        keys = ['Univariate treatment','Univariate control','Multivariate interaction','Multivariate interaction adjusted'],
                        axis=1)
    summary.to_csv(os.path.join(export_directory,'biomarker_selection_p_values.csv'))
    selected_features = summary.loc[selected_features, :]
    selected_features.to_csv(os.path.join(export_directory, 'selected_biomarkers.csv'))


# def discovery_summary(analysis_directory):
#     univariate_treatment = pd.read_csv(os.path.join(analysis_directory,'univariate_treatment.csv'), index_col=0)
#     univariate_control = pd.read_csv(os.path.join(analysis_directory, 'univariate_control.csv'), index_col=0)
#     multivariate_interaction = pd.read_csv(os.path.join(analysis_directory, 'multivariate_interaction.csv'), index_col=0)
#     multivariate_interaction_adjusted = pd.read_csv(os.path.join(analysis_directory, 'multivariate_interaction_adjusted.csv'), index_col=0)
#
#     selected_features = multivariate_interaction_adjusted.index[multivariate_interaction['Reject'] == True]
#     selected_features = multivariate_interaction_adjusted.index[multivariate_interaction_adjusted['Reject'] == True].intersection(selected_features)
#
#     summary = pd.concat([univariate_treatment.loc[selected_features, ['OR','p-value','p-value corrected']],
#                univariate_control.loc[selected_features, ['OR','p-value','p-value corrected']],
#                multivariate_interaction.loc[selected_features, ['OR','p-value','p-value corrected']],
#                multivariate_interaction_adjusted.loc[selected_features, ['OR','p-value','p-value corrected']]],
#               keys=['Treatment','Control','Interaction','Interaction adjusted'],
#               axis=1)
#     summary.to_csv(os.path.join(analysis_directory, 'summary.csv'))




if __name__ == '__main__':
    feature_directory = 'features' # directory of the folder holding all the features
    image_feature_table = pd.read_csv(os.path.join(feature_directory, 'image_features.csv'), index_col=0) # image feature table (csv) file
    clinical_feature_table = pd.read_csv(os.path.join(feature_directory, 'clinical_features.csv'), index_col=0) # clinical feature table (csv) file
    treatment = 'Treatment' # title of the treatment column in the clinical feature table
    clinical_biomarkers = ['HR','HER2','MP'] # list of clinical biomarkers in the clinical feature table used for independency testing
    outcome_title = 'pCR' # title of the treatment outcome column in the clinical feature table
    export_directory = "biomarker_discovery" # export directory of the biomarker discovery results
    image_biomarker_discovery_pipeline(image_feature_table, clinical_feature_table, treatment, clinical_biomarkers,
                                       outcome_title, export_directory, p_value_correction=False) # main function for biomarker discovery
    # discovery_summary(export_directory)
