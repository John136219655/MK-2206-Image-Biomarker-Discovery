import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import numpy as np

def patient_inclusion_analysis(clinical_record_filepath, genomics_data_filepath, proteomics_data_filepath, export_directory):
    clinical_record = pd.read_csv(clinical_record_filepath, index_col=0)
    clinical_record.index = clinical_record.index.astype(str)
    genomics_patients = pd.read_csv(genomics_data_filepath, index_col=0).T.index.astype(str)
    genomics_patients = genomics_patients.intersection(clinical_record.index)
    proteomics_patients = pd.read_csv(proteomics_data_filepath, index_col=0).index.astype(str)
    proteomics_patients = proteomics_patients.intersection(clinical_record.index)

    treatment_arm = pd.Series([0]*clinical_record.shape[0], index=clinical_record.index, name = 'Arm')
    treatment_arm[clinical_record['Arm (short name)'] == 'MK2206'] = 2
    treatment_arm[clinical_record['Arm (short name)'] == 'Ctr'] = 1

    genomics_existence = pd.Series([5]*clinical_record.shape[0], index=clinical_record.index, name = 'Genomics')
    genomics_existence[genomics_patients] = 6

    proteomics_existence = pd.Series([3] * clinical_record.shape[0], index=clinical_record.index, name = 'Proteomics')
    proteomics_existence[proteomics_patients] = 4

    inclusion_record = pd.concat([treatment_arm, proteomics_existence, genomics_existence], axis=1)
    inclusion_record = inclusion_record.sort_values(['Arm','Proteomics', 'Genomics'])
    inclusion_record.to_csv(os.path.join(export_directory, 'patient_landscape.csv'))

    my_cmap = ListedColormap(['#FFF8DE', '#E1BC29', '#C1292E', '#F2FFE3', '#679436', '#E5FFFF',
                              '#0E9594'])  # ,'#E5F3FF','#235789'])
    # set the 'bad' values (nan) to be white and transparent
    my_cmap.set_bad(color='w', alpha=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    ax.imshow(inclusion_record.T, interpolation='none', cmap=my_cmap , zorder=0,aspect='100')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(export_directory, 'patient_landscape.png'),dpi=300)
    plt.show()



def trail_simulation(treatment_probabs, control_probabs, export_directory, patient_number = 300,
                     name=None, random_draw_number=1000):
    response_diff_table = []
    success_rates = []
    for i in range(len(treatment_probabs)):
        treatment_responses = np.random.binomial(patient_number,treatment_probabs[i], size=random_draw_number)
        control_responses = np.random.binomial(patient_number,control_probabs[i], size=random_draw_number)
        response_diff = (treatment_responses-control_responses)
        # odds_ratio = treatment_responses*(patient_number-treatment_responses)/(control_responses*(patient_number-control_responses))

        response_diff_table.append(response_diff)
        success_rates.append(np.mean(response_diff>0))
    response_diff_table = pd.DataFrame(response_diff_table)
    if name is None:
        name = ''
    else:
        name = name + '_'
    mean_success_rate = np.mean(np.array(success_rates))
    # sns.kdeplot(data=success_rates)
    # plt.xlim(0,1)
    # plt.tight_layout()
    # plt.show()
    response_diff_table.to_csv(os.path.join(export_directory, name + 'vote_simulation.csv'))
    return mean_success_rate

def patient_characteristics_comparison(clinical_table, comparing_title, left_name, right_name, group_methods):
    left_data_table = clinical_table[clinical_table[comparing_title] == left_name]
    right_data_table = clinical_table[clinical_table[comparing_title] == right_name]
    summary = dict()
    for title, group_method in group_methods.items():
        if group_method == 'Mean':
            left_data = left_data_table[title].dropna()
            left_group_value = left_data.mean()
            left_min = left_data.min()
            left_max = left_data.max()
            right_data = right_data_table[title].dropna()
            right_group_value = right_data.mean()
            right_min = right_data.min()
            right_max = right_data.max()

            statistics, pvalue = mannwhitneyu(left_data_table[title].dropna().values,
                                           right_data_table[title].dropna().values)
            combined_value = pd.DataFrame([[left_group_value, right_group_value],
                                           [left_min, right_min],
                                           [left_max, right_max],
                                           [pvalue, pvalue]],
                                          index=[group_method, 'P value'], columns=[left_name, right_name])
        elif group_method == 'Median':
            left_data = left_data_table[title].dropna()
            left_group_value = left_data.median()
            left_min = left_data.min()
            left_max = left_data.max()
            right_data = right_data_table[title].dropna()
            right_group_value = right_data.median()
            right_min = right_data.min()
            right_max = right_data.max()
            statistics, pvalue = mannwhitneyu(left_data_table[title].dropna().values,
                                           right_data_table[title].dropna().values)
            combined_value = pd.DataFrame([[left_group_value, right_group_value],
                                           [left_min, right_min],
                                           [left_max, right_max],
                                           [pvalue, pvalue]],
                                          index=[group_method, 'P value'], columns=[left_name, right_name])
        elif group_method == 'Count':
            values, counts = np.unique(left_data_table[title].values.astype(str), return_counts=True)
            left_group_value = pd.Series(counts, index=values, name=left_name)
            values, counts = np.unique(right_data_table[title].values.astype(str), return_counts=True)
            right_group_value = pd.Series(counts, index=values, name=right_name)
            combined_value = pd.concat([left_group_value, right_group_value], axis=1).T.fillna(0)
            _, p, _, _ = chi2_contingency(combined_value.values)
            combined_value['P value'] = [p, p]
            combined_value = combined_value.T
        else:
            continue

        summary[title] = combined_value
    summary = pd.concat(summary, axis=0)
    return summary

def bayesian_comparison(bayesian_export_directory, subtypes, qib_subtype, export_directory):
    stats_all = []
    treatment_p_values = {}
    treatment_sc_rate = {}
    improvement_p_values = {}
    improvement_sc_rate = {}
    for subtype in subtypes:
        df = pd.read_csv(os.path.join(bayesian_export_directory, subtype, 'prediction_distributions.csv'), index_col=0)
        # get distribution
        # get bayesian summary
        Treatment_distribution = df.loc[:,'Treatment']
        Control_distribution = df.loc[:,'Control']
        ## get the mean and 95%CI
        Treatment_mean = [np.mean(Treatment_distribution)]
        Treatment_95CI = list(np.percentile(Treatment_distribution, [2.5, 97.5]))
        Control_mean = [np.mean(Control_distribution)]
        Control_95CI = list(np.percentile(Control_distribution, [2.5, 97.5]))
        # get the p-value of the comparison of two distributions
        p_value = [1-np.sum(Treatment_distribution > Control_distribution) / len(Treatment_distribution)]
        # p_value = [1-np.sum(Treatment_distribution > Control_distribution) / len(Treatment_distribution)]
        model_directory = os.path.join(bayesian_export_directory, 'models',)
        success_rate = trail_simulation(Treatment_distribution, Control_distribution, export_directory, patient_number=300,
                         name=subtype+'_treatment_vs_control_', random_draw_number=1000)
        stats = pd.Series(Treatment_mean + Treatment_95CI + Control_mean + Control_95CI + p_value+[success_rate],
                          index = ['Treatment mean','Treatment CI lower','Treatment CI higher',
                                   'Control mean','Control CI lower','Control CI higher',
                                   'p-value','Successful rate'], name = subtype)
        stats_all.append(stats)

        if subtype == 'All':
            new_subtype = qib_subtype
        else:
            new_subtype = subtype+qib_subtype
        df = pd.read_csv(os.path.join(bayesian_export_directory, new_subtype, 'prediction_distributions.csv'),
                         index_col=0)
        # get distribution
        # get bayesian summary
        Treatment_distribution_qib = df.loc[:, 'Treatment']
        Control_distribution_qib = df.loc[:, 'Control']
        ## get the mean and 95%CI
        Treatment_mean = [np.mean(Treatment_distribution_qib)]
        Treatment_95CI = list(np.percentile(Treatment_distribution_qib, [2.5, 97.5]))
        Control_mean = [np.mean(Control_distribution_qib)]
        Control_95CI = list(np.percentile(Control_distribution_qib, [2.5, 97.5]))
        # get the p-value of the comparison of two distributions
        success_rate = trail_simulation(Treatment_distribution_qib, Control_distribution_qib, export_directory,
                                        patient_number=300,
                                        name=subtype + '_treatment_vs_control_', random_draw_number=1000)
        p_value = [
            1 - np.sum(Treatment_distribution_qib > (Control_distribution_qib)) / len(Treatment_distribution_qib)]
        # p_value = [1-np.sum(Treatment_distribution > Control_distribution) / len(Treatment_distribution)]
        stats = pd.Series(Treatment_mean + Treatment_95CI + Control_mean + Control_95CI + p_value+[success_rate],
                          index=['Treatment mean', 'Treatment CI lower', 'Treatment CI higher',
                                 'Control mean', 'Control CI lower', 'Control CI higher',
                                 'p-value', 'Successful rate'], name = new_subtype)
        stats_all.append(stats)

        success_rate = trail_simulation(Treatment_distribution_qib, Treatment_distribution, export_directory,
                                        patient_number=300,
                                        name=subtype + '_qib_treatment_', random_draw_number=1000)
        p_value = 1 - np.sum(Treatment_distribution_qib > Treatment_distribution) / len(
            Treatment_distribution)
        treatment_p_values[subtype] = p_value
        treatment_sc_rate[subtype] = success_rate

        p_value = 1 - np.sum((Treatment_distribution_qib-Control_distribution_qib) > (Treatment_distribution-Control_distribution)) / len(
            Treatment_distribution)
        improvement_p_values[subtype] = p_value
    stats_all = pd.concat(stats_all, axis=1).T
    treatment_p_values = pd.Series(treatment_p_values, name = 'Treatment p-value')
    improvement_p_values = pd.Series(improvement_p_values, name = 'Improvement p-value')
    treatment_sc_rate = pd.Series(treatment_sc_rate, name='Treatment successful rate')
    stats_all = pd.concat([stats_all, treatment_p_values, improvement_p_values, treatment_sc_rate], axis=1)
    stats_all.to_csv(os.path.join(export_directory, qib_subtype+'_bayesian_comparison_stats.csv'))



if __name__ == '__main__':

    feature_directory = "features"
    export_directory = "statistical_analysis"
    clinical_feature_table = pd.read_csv(os.path.join(feature_directory, 'clinical_features.csv'), index_col=0)
    menopausal_status = []
    for patient_id in clinical_feature_table.index:
        value = clinical_feature_table.loc[patient_id,'menopausal_status']
        if value != value:
            menopausal_status.append(None)
            continue
        standardized_value = None
        if 'Perimenopausal' in value:
            standardized_value = 'Perimenopausal'
        elif 'Postmenopausal' in value:
            standardized_value = 'Postmenopausal'
        elif 'Premenopausal' in value:
            standardized_value = 'Premenopausal'
        elif 'not applicable' in value:
            standardized_value = 'N/A'
        menopausal_status.append(standardized_value)
    clinical_feature_table['menopausal_status'] = menopausal_status

    comparing_title = 'Treatment'
    left_name = 0
    right_name = 1
    group_methods = {
        'HR': 'Count',
        'HER2': 'Count',
        'MP': 'Count',
        'Race': 'Count',
        'menopausal_status': 'Count',
        'ethnicity':'Count',
        'pCR': 'Count'
    }
    summary = patient_characteristics_comparison(clinical_feature_table, comparing_title, left_name, right_name, group_methods)
    summary.to_csv(os.path.join(export_directory, 'patient_characteristics_comparison_treatment.csv'))


    subtypes = [
        'All',
        'HER2-',
        'HER2+',
        'HR-',
        'HR+',
        'HR-HER2-',
        'HR-HER2+',
        'HR+HER2-',
        'HR+HER2+',
        'MP-',
        'MP+'

    ]

    qib_subtype = 'GLCM_SS+'
    bayesian_export_directory = r"biomarker_assessment\GLCM_SS\bayesian"
    export_directory = "statistical_analysis"
    bayesian_comparison(bayesian_export_directory, subtypes, qib_subtype, export_directory)
    #
