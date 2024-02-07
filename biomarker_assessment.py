
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import os
import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import fisher_exact, norm
import pymc as pm
import arviz as az
import pytensor.tensor as at
import matplotlib.pyplot as plt
import seaborn as sns

def threshold_optimization(values, treatment, outcome):
    efficacies = []
    thresholds = []
    auc = roc_auc_score(outcome[treatment], values[treatment])

    for threshold in np.sort(values):
        if auc > 0.5:
            interaction = np.all([treatment, values > threshold], axis=0)
        else:
            interaction = np.all([treatment, ~(values > threshold)], axis=0)
        if np.sum(interaction) == 0:
            continue
        if np.sum(~interaction) == 0:
            continue
        # if len(outcome_values) == 0:
        #     continue
        # if len(~outcome_values) == 0:
        #     continue
        try:
            table = pd.crosstab(interaction, outcome)
            oddsratio, pvalue = fisher_exact(table)
            efficacies.append(1/pvalue)
            thresholds.append(threshold)
        except:
            continue
    threshold = thresholds[np.argmax(efficacies)]
    # fpr, tpr, thresholds = roc_curve(outcome, values)
    # max_index = np.argmax(tpr-fpr)
    # threshold = thresholds[max_index]
    return threshold


def pcr_rate_logistics(alpha, beta, gamma, delta, feature_table, treatment_names, biomarker_names, interaction_names):
    prediction = alpha + at.dot(feature_table[treatment_names].values, beta) + at.dot(feature_table[biomarker_names].values, gamma) + at.dot(feature_table[interaction_names].values, delta)
    # if len(treatment_names) > 0:
    #     prediction += at.dot(feature_table[treatment_names].values, beta)
    # if len(biomarker_names) > 0:
    #     prediction += at.dot(feature_table[biomarker_names].values, gamma)
    # if len(interaction_names) > 0:
    #     prediction += at.dot(feature_table[interaction_names].values, delta)
    return pm.math.invlogit(prediction)

def model_training(feature_table, treatment_names, biomarker_names,interaction_names, prediction_name,
                   export_directory):
    with pm.Model(coords={"treatments": treatment_names,
                                   "biomarkers": biomarker_names,
                                   'interactions': interaction_names}) as basic_model:
        # Priors for unknown model parameters
        alpha = pm.Uniform("interception", lower=-10, upper=10)
        beta = pm.Uniform("treatment", lower=-10, upper=10, dims='treatments')
        gamma = pm.Uniform("biomarker", lower=-10, upper=10, dims='biomarkers')
        delta = pm.Normal("interaction", mu=1, sigma=1, dims='interactions')

        # pm.Deterministic()

        # Expected value of outcome
        prob = pcr_rate_logistics(alpha, beta, gamma, delta, feature_table, treatment_names, biomarker_names,
                                  interaction_names)

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Binomial("Y_obs", n=1, p=prob, observed=feature_table[prediction_name].values)

        idata = pm.sample(model = basic_model, draws=1000, tune=2000, cores=1, chains=4, return_inferencedata=True, compute_convergence_checks=False,
                          jitter_max_retries=50)
        # idata = pm.sample_posterior_predictive(idata, return_inferencedata=True, extend_inferencedata=True)

        try:
            az.plot_trace(idata)
            plt.tight_layout()
            plt.savefig(os.path.join(export_directory,'trace.pdf'))
            plt.show()
        except:
            pass

        summary = az.summary(idata, round_to=2)
        print(summary)
        summary.to_csv(os.path.join(export_directory,'summary.csv'))

        # posterior_dataframe = [idata_predictions.posterior['interception'].to_dataframe(),
        #                        idata_predictions.posterior['treatment'].to_dataframe(),
        #                        idata_predictions.posterior['biomarker'].to_dataframe(),
        #                        idata_predictions.posterior['interaction'].to_dataframe()]
        # posterior_dataframe = pd.concat(posterior_dataframe, axis=1)
        # posterior_dataframe.to_csv(os.path.join(export_directory, 'posterior_interception.csv'))
        posterior_dataframe = [
            idata.posterior['interception'].to_dataframe(),
            idata.posterior['treatment'].to_dataframe().unstack(level=-1).droplevel(0, axis=1),
            idata.posterior['biomarker'].to_dataframe().unstack(level=-1).droplevel(0, axis=1),
            idata.posterior['interaction'].to_dataframe().unstack(level=-1).droplevel(0, axis=1)
        ]
        posterior_dataframe = pd.concat(posterior_dataframe, axis=1)
        posterior_dataframe.to_csv(os.path.join(export_directory, 'posterior_distributions.csv'))
        # coefficients = []
        # coefficients.append(pd.Series(np.array(idata.posterior['interception']).flatten(), name='interception'))
        # beta_distribution = np.array(idata.posterior['treatment'])
        # for i in range(beta_distribution.shape[-1]):
        #     coefficients.append(pd.Series(beta_distribution[:, :, i].flatten(), name=treatment_names[i]))
        # gamma_distribution = np.array(idata.posterior['biomarker'])
        # for i in range(gamma_distribution.shape[-1]):
        #     coefficients.append(pd.Series(gamma_distribution[:, :, i].flatten(), name=biomarker_names[i]))
        # delta_distribution = np.array(idata.posterior['interaction'])
        # for i in range(delta_distribution.shape[-1]):
        #     coefficients.append(pd.Series(delta_distribution[:, :, i].flatten(), name=interaction_names[i]))
        # coefficients = pd.concat(coefficients, axis=1)
        # coefficients.to_csv(os.path.join(export_directory,'coefficients.csv'))

def prediction_distribution(model_directory, positive_characteristics, export_directory):
    model_parameters = pd.read_csv(os.path.join(model_directory, 'posterior_distributions.csv'), index_col=[0,1])
    distributions = []
    for title, positive_characteristic in positive_characteristics.items():
        selected_parameters = model_parameters[list(positive_characteristic)+['interception']]
        distribution = 1/(1+np.exp(-np.sum(selected_parameters.values, axis=1)))
        distributions.append(pd.Series(distribution, name=title))
    distributions = pd.concat(distributions, axis=1)
    distributions.to_csv(os.path.join(export_directory, 'prediction_distributions.csv'))
    # plt.figure(figsize=figure_size)
    # sns.kdeplot(data=distributions)
    # plt.xlabel(x_title)
    # plt.title(figure_title)
    # plt.xlim(0,1)
    # plt.tight_layout()
    # plt.savefig(os.path.join(export_directory, name+'prediction_distributions.png'), dpi=300)
    # plt.show()

def two_group_binomial_power(treatment_probab, control_probab, patient_number=150):
    prob_diff = treatment_probab-control_probab
    average_prob = (treatment_probab+control_probab)/2
    a = np.sqrt(treatment_probab*(1-treatment_probab)/patient_number+control_probab*(1-control_probab)/patient_number)
    b = np.sqrt(average_prob*(1-average_prob)*(2/patient_number))
    z_value = (prob_diff-1.96*b)/a
    p = norm.cdf(z_value)
    return p

def trail_simulation(treatment_probabs, control_probabs, patient_number = 300):
    success_rates = []
    for i in range(len(treatment_probabs)):
        success_rates.append(two_group_binomial_power(treatment_probabs[i],control_probabs[i], patient_number=patient_number))
    average_success_rate = np.mean(np.array(success_rates))
    return average_success_rate

def power_estimation(p1, p2, alpha, sample_size, random_draw_number=1000):
    treatment_responses = (np.random.uniform(0,1,size=random_draw_number)*sample_size).astype(int)
    control_responses = (np.random.uniform(0,1,size=random_draw_number)*sample_size).astype(int)
    odds_ratio_dfr = treatment_responses * (sample_size - treatment_responses) / (
            control_responses * (sample_size - control_responses))
    print(np.mean(odds_ratio_dfr))
    print(np.std(odds_ratio_dfr))
    sns.histplot(odds_ratio_dfr)
    plt.show()

    # critical_value = scipy.stats.norm.ppf((1-alpha), loc=1, scale=p2)
    critical_value = np.percentile(odds_ratio_dfr, (1-alpha)*100)
    treatment_responses = np.random.binomial(sample_size, p1, size=random_draw_number)
    control_responses = np.random.binomial(sample_size, p2, size=random_draw_number)
    odds_ratio = treatment_responses * (sample_size - treatment_responses) / (
            control_responses * (sample_size - control_responses))
    sns.histplot(odds_ratio)
    plt.show()
    print(np.mean(odds_ratio > critical_value))

def posterior_distribution_comparison(data_directory,x_title='',figure_size = (3,3),
                            figure_title=''):
    prediction_distributions = pd.read_csv(os.path.join(data_directory, 'prediction_distributions.csv'), index_col=0)[['Control','Treatment']]
    # plt.figure(figsize=figure_size)
    # sns.kdeplot(data=prediction_distributions, palette={'Control': 'steelblue', 'Treatment': 'firebrick'})
    # plt.xlabel(x_title)
    # plt.title(figure_title)
    # plt.xlim(0, 1)
    # plt.tight_layout()
    # plt.savefig(os.path.join(data_directory, 'prediction_distributions.png'), dpi=300)
    # plt.show()
    means = prediction_distributions.mean(axis=0)
    means.index = [x+' mean' for x in means.index]
    lower_cis = prediction_distributions.quantile(0.05, axis=0)
    lower_cis.index = [x+' lower CI' for x in lower_cis.index]
    higher_cis = prediction_distributions.quantile(0.95, axis=0)
    higher_cis.index = [x+' higher CI' for x in higher_cis.index]
    statistics = pd.concat([means, lower_cis, higher_cis], axis=0)
    effective_proba = prediction_distributions['Treatment'].values > prediction_distributions['Control'].values
    effective_proba = np.mean(effective_proba)
    statistics['Higher pCR probability'] = effective_proba
    success_rate = trail_simulation(prediction_distributions['Treatment'].values, prediction_distributions['Control'].values,
                     patient_number=150)
    statistics['Trail success rate'] = success_rate
    return statistics

def bayesian_analysis(image_feature_table, clinical_feature_table, image_biomarker, clinical_biomarkers,positive_biomarker_list,
                           treatment, outcome, export_directory):
    if image_biomarker is not None:
        combined_table = pd.concat([image_feature_table[image_biomarker], clinical_feature_table[clinical_biomarkers+[treatment,outcome]]], axis=1).dropna()
        biomarkers = clinical_biomarkers+[image_biomarker]
    else:
        combined_table = clinical_feature_table[clinical_biomarkers + [treatment, outcome]]
        biomarkers = clinical_biomarkers
    for feature in biomarkers:
        combined_table[feature + '_interaction'] = combined_table[treatment]*combined_table[feature]
    treatment_names = ['Treatment']
    interaction_names = [feature + '_interaction' for feature in biomarkers]
    print('Building model for biomarkers {0}, interactions {1}, and treatment'.format(biomarkers, interaction_names))
    if len(biomarkers) == 0:
        model_export_directory = os.path.join(export_directory, 'models', 'All')
    else:
        model_export_directory = os.path.join(export_directory, 'models','_'.join(biomarkers))
    if not os.path.exists(model_export_directory):
        os.makedirs(model_export_directory)
        model_training(combined_table, treatment_names, biomarkers, interaction_names, outcome,
                       model_export_directory)
    results = []
    if len(biomarkers) == 0:
        positive_characteristics = {
            'Treatment': ['Treatment'],
            'Control': []
        }
        name = 'All'
        distribution_export_directory = os.path.join(export_directory, name)
        if not os.path.exists(distribution_export_directory):
            os.mkdir(distribution_export_directory)
        print('Generating predictions for signature {0}'.format(name))
        prediction_distribution(model_export_directory, positive_characteristics, distribution_export_directory)
        statistics = posterior_distribution_comparison(distribution_export_directory)
        statistics.name = name
        results.append(statistics)
    else:
        for positive_biomarkers in positive_biomarker_list:
            positive_characteristics = {
                'Treatment': ['Treatment']+positive_biomarkers+[x+'_interaction' for x in positive_biomarkers],
                'Control': positive_biomarkers
            }
            name = ''.join([biomarker+'+' if biomarker in positive_biomarkers else biomarker + '-' for biomarker in biomarkers])
            distribution_export_directory = os.path.join(export_directory, name)
            if not os.path.exists(distribution_export_directory):
                os.mkdir(distribution_export_directory)
            print('Generating predictions for signature {0}'.format(name))
            prediction_distribution(model_export_directory, positive_characteristics, distribution_export_directory)
            statistics = posterior_distribution_comparison(distribution_export_directory)
            statistics.name = name
            results.append(statistics)
    results = pd.concat(results, axis=1).T
    return results

def bayesian_analysis_expansion(image_feature_table, clinical_feature_table, name, biomarkers, positive_status,
                           treatment, outcome, export_directory):
    combined_table = pd.concat([image_feature_table, clinical_feature_table],axis=1)[biomarkers+[treatment, outcome]].dropna()
    positive_results = None
    for biomarker, status in zip(biomarkers, positive_status):
        values = combined_table[biomarker]
        if not status:
            values = 1-values
        if positive_results is None:
            positive_results = values
        else:
            positive_results = positive_results + values
    combined_table[name] = (positive_results>0).astype(int)
    combined_table[name + '_interaction'] = combined_table[treatment] * combined_table[name]
    print('Building model for biomarkers {0}, interactions {1}, and treatment'.format(name, name + '_interaction'))
    # pcr_rate = combined_table[combined_table[name] == 1][outcome]
    pcr_rate = combined_table[combined_table[name] == 1]
    treatment_pcr_rate = pcr_rate[pcr_rate[treatment] == 1][outcome]
    control_pcr_rate = pcr_rate[pcr_rate[treatment] == 0][outcome]
    treatment_pcr_rate = treatment_pcr_rate.sum()/treatment_pcr_rate.shape[0]
    control_pcr_rate = control_pcr_rate.sum()/control_pcr_rate.shape[0]
    print('Raw pCR rate: {0} vs. {1}'.format(treatment_pcr_rate, control_pcr_rate))
    # model_export_directory = os.path.join(export_directory, 'models', name)
    # if not os.path.exists(model_export_directory):
    #     os.makedirs(model_export_directory)
    # treatment_names = ['Treatment']
    # interaction_names = [name + '_interaction']
    # model_training(combined_table, treatment_names, [name], interaction_names, outcome,
    #                model_export_directory)
    #
    # positive_characteristics = {
    #     'Treatment': ['Treatment', name, name + '_interaction'],
    #     'Control': [name]
    # }
    # distribution_export_directory = os.path.join(export_directory, name)
    # if not os.path.exists(distribution_export_directory):
    #     os.mkdir(distribution_export_directory)
    # print('Generating predictions for signature {0}'.format(name))
    # prediction_distribution(model_export_directory, positive_characteristics, distribution_export_directory)
    # statistics = posterior_distribution_comparison(distribution_export_directory)
    # statistics.name = name
    # return statistics



def bayesian_posterior_combination(signatures, prevalence_table, export_directory):
    combined_distributions = []
    prevalence_list = []
    for prevalence_signature, signature in signatures:
        pcr_predictions = pd.read_csv(os.path.join(export_directory,signature,
                                                    'prediction_distributions.csv'), index_col=0)
        prevalence = prevalence_table[prevalence_signature]
        prevalence_list.append(prevalence)
        combined_distributions.append(pcr_predictions)
    prevalence_list = np.array(prevalence_list)
    prevalence_list = prevalence_list/np.max(prevalence_list)*4000
    prevalence_list = prevalence_list.astype(int)
    for i, prevalence in enumerate(prevalence_list):
        combined_distributions[i] = combined_distributions[i].loc[np.random.choice(combined_distributions[i].index.values, prevalence, replace=True)]

    combined_distributions = pd.concat(combined_distributions, axis=0, ignore_index=True)
    plt.figure(figsize=(3,3))
    sns.kdeplot(data=combined_distributions, palette={'Control': 'steelblue', 'Treatment': 'firebrick'})
    # plt.xlabel(x_title)
    # plt.title(figure_title)
    plt.xlim(0, 1)
    plt.tight_layout()
    # plt.savefig(os.path.join(export_directory,'prediction_distributions.png'), dpi=300)
    plt.show()
    means = combined_distributions.mean(axis=0)
    means.index = [x + ' mean' for x in means.index]
    lower_cis = combined_distributions.quantile(0.05, axis=0)
    lower_cis.index = [x + ' lower CI' for x in lower_cis.index]
    higher_cis = combined_distributions.quantile(0.95, axis=0)
    higher_cis.index = [x + ' higher CI' for x in higher_cis.index]
    statistics = pd.concat([means, lower_cis, higher_cis], axis=0)
    effective_proba = combined_distributions['Treatment'].values > combined_distributions['Control'].values
    effective_proba = np.mean(effective_proba)
    statistics['Higher pCR probability'] = effective_proba
    success_rate = trail_simulation(combined_distributions['Treatment'].values,
                                    combined_distributions['Control'].values,
                                    patient_number=150)
    statistics['Trail success rate'] = success_rate
    return statistics

    




def continuous_performance(image_feature_table, clinical_feature_table, feature_icc, image_biomarkers, clinical_biomarkers,
                           treatment, outcome, export_directory):
    variances = {}
    for image_feature in image_biomarkers:
        variance = image_feature_table[image_feature].var()
        variances[image_feature] = variance
    variances = pd.Series(variances, name = 'Variance')

    iccs = feature_icc.loc[image_biomarkers, :]

    treatment_clinical = clinical_feature_table[clinical_feature_table[treatment] == 1]
    treatment_image_features = image_feature_table.loc[treatment_clinical.index,image_biomarkers]
    treatment_univariate_aucs = {}
    treatment_multivariate_aucs = {}
    for image_feature in image_biomarkers:
        auc = roc_auc_score(treatment_clinical[outcome],treatment_image_features[image_feature])
        # if auc < 0.5:
        #     auc = 1-auc
        treatment_univariate_aucs[image_feature] = auc

        model_descriptions = outcome + ' ~ ' + ' + '.join([image_feature]+clinical_biomarkers)
        model_data = pd.concat([treatment_image_features[image_feature],treatment_clinical[[outcome]+clinical_biomarkers]], axis=1).dropna()
        model = smf.logit(model_descriptions, data=model_data).fit()
        auc = roc_auc_score(treatment_clinical[outcome], model.predict(model_data))
        treatment_multivariate_aucs[image_feature] = auc
        

    treatment_univariate_aucs = pd.Series(treatment_univariate_aucs, name = 'Treatment AUC univariate')
    treatment_multivariate_aucs = pd.Series(treatment_multivariate_aucs, name = 'Treatment AUC multivariate')

    control_clinical = clinical_feature_table[clinical_feature_table[treatment] == 0]
    control_image_features = image_feature_table.loc[control_clinical.index, image_biomarkers]
    control_univariate_aucs = {}
    control_multivariate_aucs = {}
    for image_feature in image_biomarkers:
        auc = roc_auc_score(control_clinical[outcome], control_image_features[image_feature])
        # if auc < 0.5:
        #     auc = 1-auc
        control_univariate_aucs[image_feature] = auc

        model_descriptions = outcome + ' ~ ' + ' + '.join([image_feature]+clinical_biomarkers)
        model_data = pd.concat(
            [control_image_features[image_feature], control_clinical[[outcome] + clinical_biomarkers]],
            axis=1).dropna()
        model = smf.logit(model_descriptions, data=model_data).fit()
        auc = roc_auc_score(control_clinical[outcome], model.predict(model_data))
        control_multivariate_aucs[image_feature] = auc

    control_univariate_aucs = pd.Series(control_univariate_aucs, name='Control AUC univariate')
    control_multivariate_aucs = pd.Series(control_multivariate_aucs, name = 'Control AUC multivariate')


    results = pd.concat([variances, iccs, treatment_univariate_aucs, control_univariate_aucs, treatment_multivariate_aucs, control_multivariate_aucs], axis=1)
    results.to_csv(os.path.join(export_directory, 'univariate_performance.csv'))
        
def pcr_association(features, outcomes):
    table = pd.crosstab(features.values, outcomes.values)
    print(table)
    # contigency_table = Table2x2(table)
    # odds = contigency_table.oddsratio
    oddsratio, pvalue = fisher_exact(table)
    stats = pd.Series([oddsratio, pvalue],index=['OR','p-value'])
    return stats

# def pcr_association(data, treatment, outcome):
#     model_descriptions = outcome + ' ~ ' + treatment
#     model = smf.logit(model_descriptions, data=data).fit()
#     result = pd.Series(
#         {
#             "OR": np.exp(model.params[treatment]),
#             "Lower CI": np.exp(model.conf_int()[0][treatment]),
#             "Upper CI": np.exp(model.conf_int()[1][treatment]),
#             'p-value': model.pvalues[treatment]
#         }
#     )
#     return result

def subtype_treatment_pcr_association(image_feature_table, clinical_feature_table, image_biomarker, clinical_subtypes,
                                      treatment, outcome, export_directory):
    combined_stats = []
    for clinical_subtype in clinical_subtypes:
        if clinical_subtype is None:
            selected_clinical_feature_table = clinical_feature_table
            subtype_name = 'All'
        else:
            selected_clinical_feature_table = clinical_feature_table.query(' and '.join([name+' == '+str(value) for name, value in clinical_subtype]))
            subtype_name = '/'.join([name+'+' if value == 1 else name+'-' for name, value in clinical_subtype])
        all_stats = pcr_association(selected_clinical_feature_table[treatment], selected_clinical_feature_table[outcome])
        all_stats['Patient Num'] = selected_clinical_feature_table.shape[0]
        all_stats['Patient Ratio'] = (selected_clinical_feature_table.shape[0] / clinical_feature_table.shape[0])
        subgroup_positive_index = image_feature_table.index[image_feature_table[image_biomarker] == 1].intersection(selected_clinical_feature_table.index)
        print(subgroup_positive_index)
        subgroup_positive_stats = pcr_association(selected_clinical_feature_table.loc[subgroup_positive_index,treatment], 
                                         selected_clinical_feature_table.loc[subgroup_positive_index,outcome])
        subgroup_positive_stats['Patient Num'] = len(subgroup_positive_index)
        subgroup_positive_stats['Patient Ratio'] = (len(subgroup_positive_index) / clinical_feature_table.shape[0])
        subgroup_negative_index = image_feature_table.index[image_feature_table[image_biomarker] == 0].intersection(selected_clinical_feature_table.index)
        subgroup_negative_stats = pcr_association(selected_clinical_feature_table.loc[subgroup_negative_index, treatment],
                                         selected_clinical_feature_table.loc[subgroup_negative_index, outcome])
        subgroup_negative_stats['Patient Num'] = len(subgroup_negative_index)
        subgroup_negative_stats['Patient Ratio'] = (len(subgroup_negative_index) / clinical_feature_table.shape[0])
        expanded_positive_index = image_feature_table.index[image_feature_table[image_biomarker] == 1].union(selected_clinical_feature_table.index)
        expanded_positive_stats = pcr_association(clinical_feature_table.loc[expanded_positive_index, treatment],
                                         clinical_feature_table.loc[expanded_positive_index, outcome])
        expanded_positive_stats['Patient Num'] = len(expanded_positive_index)
        expanded_positive_stats['Patient Ratio'] = (len(expanded_positive_index) / clinical_feature_table.shape[0])
        expanded_negative_index = image_feature_table.index[image_feature_table[image_biomarker] == 0].union(selected_clinical_feature_table.index)
        expanded_negative_stats = pcr_association(
            clinical_feature_table.loc[expanded_negative_index, treatment],
            clinical_feature_table.loc[expanded_negative_index, outcome])
        expanded_negative_stats['Patient Num'] = len(expanded_negative_index)
        expanded_negative_stats['Patient Ratio'] = (len(expanded_negative_index) / clinical_feature_table.shape[0])
        stats = pd.concat([all_stats, subgroup_positive_stats,subgroup_negative_stats,
                           expanded_positive_stats, expanded_negative_stats], axis=1,
                          keys = [subtype_name,subtype_name+'/'+image_biomarker+'+',
                                  subtype_name+'/'+image_biomarker+'-', subtype_name+' plus '+image_biomarker+'+',
                                  subtype_name+' plus '+image_biomarker+'-'])
        combined_stats.append(stats)
    combined_stats = pd.concat(combined_stats, axis=1).T
    combined_stats.to_csv(os.path.join(export_directory, 'subgroup_treatment_efficacy.csv'))

def bayesian_analysis_pipeline(image_feature_table, clinical_feature_table, image_biomarker, clinical_biomarkers,
                               treatment, outcome_title, export_directory):
    combined_results = []
    positive_biomarker_list = [[]]
    results = bayesian_analysis(image_feature_table, clinical_feature_table, None, [],
                                positive_biomarker_list,
                                treatment, outcome_title, export_directory)
    combined_results.append(results)

    positive_biomarker_list = [[image_biomarker], []]
    results = bayesian_analysis(image_feature_table, clinical_feature_table, image_biomarker, [],
                                positive_biomarker_list,
                                treatment, outcome_title, export_directory)
    combined_results.append(results)

    for clinical_biomarker in clinical_biomarkers:
        positive_biomarker_list = [[clinical_biomarker],[]]
        results = bayesian_analysis(image_feature_table, clinical_feature_table,None, [clinical_biomarker],
                          positive_biomarker_list,
                          treatment, outcome_title, export_directory)
        combined_results.append(results)
        positive_biomarker_list = [[clinical_biomarker, image_biomarker],[image_biomarker],[clinical_biomarker],[]]
        results = bayesian_analysis(image_feature_table, clinical_feature_table, image_biomarker, [clinical_biomarker],
                          positive_biomarker_list,
                          treatment, outcome_title, export_directory)
        combined_results.append(results)

    positive_biomarker_list = [
        ['HR'],
        ['HER2'],
        [],
        ['HR','HER2']
    ]
    results = bayesian_analysis(image_feature_table, clinical_feature_table, None, ['HR','HER2'],
                      positive_biomarker_list,
                      treatment, outcome_title, export_directory)
    combined_results.append(results)
    positive_biomarker_list = [
        ['HR', image_biomarker],
        ['HER2',image_biomarker],
        [image_biomarker],
        ['HR', 'HER2',image_biomarker],
        ['HR'],
        ['HER2'],
        [],
        ['HR', 'HER2']
    ]
    results = bayesian_analysis(image_feature_table, clinical_feature_table, image_biomarker, ['HR', 'HER2'],
                      positive_biomarker_list,
                      treatment, outcome_title, export_directory)
    combined_results.append(results)
    combined_results = pd.concat(combined_results, axis=0)
    combined_results.to_csv(os.path.join(export_directory,'subgroup_results.csv'))




def biomarker_assessment_pipeline(data_directory, feature_name_mapping, clinical_subtypes, treatment,clinical_biomarkers,
                                  outcome_title, final_biomarker):
    biomarker_discovery_directory = os.path.join(data_directory, 'biomarker_discovery')
    feature_directory = os.path.join(data_directory, 'features')
    export_directory = os.path.join(data_directory, "biomarker_assessment")
    if not os.path.exists(export_directory):
        os.mkdir(export_directory)
    final_biomarker_export_directory = os.path.join(export_directory, final_biomarker)
    if not os.path.exists(final_biomarker_export_directory):
        os.mkdir(final_biomarker_export_directory)
    clinical_feature_table = pd.read_csv(os.path.join(feature_directory, 'clinical_features.csv'), index_col=0)

    image_feature_table = pd.read_csv(os.path.join(feature_directory, 'image_features.csv'), index_col=0)
    binarized_image_feature_table = pd.read_csv(os.path.join(feature_directory, 'image_features_binarized.csv'),
                                                index_col=0)
    image_feature_table = image_feature_table.rename(feature_name_mapping, axis=1)
    binarized_image_feature_table = binarized_image_feature_table.rename(feature_name_mapping, axis=1)


    feature_icc = pd.read_csv(os.path.join(feature_directory, 'iccs.csv'), index_col=0).rename(feature_name_mapping, axis=0)


    image_biomarker_candidates = pd.read_csv(os.path.join(biomarker_discovery_directory, 'selected_biomarkers.csv'),
                                   index_col=0, header=[0, 1]).rename(feature_name_mapping, axis=0).index
    continuous_performance(image_feature_table, clinical_feature_table, feature_icc, image_biomarker_candidates, clinical_biomarkers, treatment, outcome_title,
                           export_directory)
    subtype_treatment_pcr_association(binarized_image_feature_table, clinical_feature_table, final_biomarker, clinical_subtypes,
                                      treatment, outcome_title, final_biomarker_export_directory)

    bayesian_export_directory = os.path.join(final_biomarker_export_directory, 'bayesian')
    if not os.path.exists(bayesian_export_directory):
        os.mkdir(bayesian_export_directory)

    bayesian_analysis_pipeline(binarized_image_feature_table, clinical_feature_table, final_biomarker, clinical_biomarkers,
                               treatment, outcome_title, bayesian_export_directory)




if __name__ == '__main__':
    data_directory = ''
    feature_name_mapping = {
        'log_sigma_3_mm_3D_firstorder_Entropy_32_binCount': 'Entropy',
        'log_sigma_3_mm_3D_firstorder_Uniformity_32_binCount': 'Uniformity',
        'log_sigma_3_mm_3D_glcm_JointEntropy_32_binCount': 'GLCM_JE',
        'log_sigma_3_mm_3D_glcm_SumEntropy_32_binCount': 'GLCM_SE',
        'log_sigma_3_mm_3D_glcm_SumSquares_32_binCount': 'GLCM_SS',
        'log_sigma_3_mm_3D_glrlm_GrayLevelNonUniformityNormalized_32_binCount': 'GLRLM_GLNUN',
        'log_sigma_3_mm_3D_glszm_GrayLevelNonUniformityNormalized_32_binCount': 'GLSZM_GLNUN'
    }
    clinical_subtypes = [
        None,
        [('HR', 0)],
        [('HR', 1)],
        [('HER2', 0)],
        [('HER2', 1)],
        [('HR', 0), ('HER2', 0)],
        [('HR', 0), ('HER2', 1)],
        [('HR', 1), ('HER2', 0)],
        [('HR', 1), ('HER2', 1)],
        [('MP', 0)],
        [('MP', 1)]
    ]
    treatment = 'Treatment'
    clinical_biomarkers = ['HR', 'HER2', 'MP']
    outcome_title = 'pCR'

    final_biomarker = 'GLCM_SS'
    biomarker_assessment_pipeline(data_directory, feature_name_mapping, clinical_subtypes, treatment,
                                  clinical_biomarkers,
                                  outcome_title, final_biomarker)




