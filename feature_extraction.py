
import os
import pandas as pd
from radiomics.featureextractor import RadiomicsFeatureExtractor

def radiomics_feature_extraction(database_directory, time_information, mask_name, feature_extraction_directory):
    parameter_filepath = os.path.join(feature_extraction_directory,'feature_extraction_parameters.yaml')
    feature_extractor = RadiomicsFeatureExtractor(parameter_filepath)
    feature_table = {}
    for patient_id in os.listdir(database_directory):
        patient_folder = os.path.join(database_directory, patient_id)
        # get the phase index for the early contrast phase
        phase_index = time_information.loc[patient_id, 'early_contrast_index']
        image_name = 'DCE_MRIT0_Phase_{0}.mha'.format(phase_index)
        image_filepath = os.path.join(patient_folder, image_name)
        mask_filepath = os.path.join(patient_folder, mask_name)
        feature_values = feature_extractor.execute(image_filepath, mask_filepath)
        feature_table[patient_id] = pd.Series(feature_values)
    feature_table = pd.concat(feature_table, axis=1).T
    feature_table.columns = [x.replace('-', '_') for x in feature_table.columns]
    feature_table.to_csv(os.path.join(feature_extraction_directory, 'image_features.csv'))

def radiomics_feature_extraction_with_normalization(database_directory, time_information, mask_name, feature_extraction_directory):
    parameter_filepath = os.path.join(feature_extraction_directory,'feature_extraction_parameters_with_normalization.yaml')
    feature_extractor = RadiomicsFeatureExtractor(parameter_filepath)
    feature_table = {}
    for patient_id in os.listdir(database_directory):
        patient_folder = os.path.join(database_directory, patient_id)
        # get the phase index for the early contrast phase
        phase_index = time_information.loc[patient_id, 'early_contrast_index']
        image_name = 'DCE_MRIT0_Phase_{0}.mha'.format(phase_index)
        image_filepath = os.path.join(patient_folder, image_name)
        mask_filepath = os.path.join(patient_folder, mask_name)
        feature_values = feature_extractor.execute(image_filepath, mask_filepath)
        feature_table[patient_id] = pd.Series(feature_values)
    feature_table = pd.concat(feature_table, axis=1).T
    feature_table.columns = [x.replace('-', '_') for x in feature_table.columns]
    feature_table.to_csv(os.path.join(feature_extraction_directory, 'image_features_with_normalization.csv'))

def normalization_comparison(feature_name,feature_extraction_directory):
    feature_values_without_norm = pd.read_csv(os.path.join(feature_extraction_directory, 'image_features.csv'), index_col=0)
    feature_values_without_norm = feature_values_without_norm[feature_name]
    feature_values_with_norm = pd.read_csv(os.path.join(feature_extraction_directory, 'image_features_with_normalization.csv'), index_col=0)
    feature_values_with_norm = feature_values_with_norm[feature_name]
    # match patient ids and drop missing values
    feature_values = pd.concat([feature_values_with_norm, feature_values_without_norm], axis=1).dropna()
    feature_value_diffs = feature_values.iloc[:,0].values-feature_values.iloc[:,1].values
    print(feature_value_diffs)



if __name__ == '__main__':
    database_directory = 'demo_image_dataset'
    feature_extraction_directory = 'feature_extraction'
    time_information = pd.read_csv(os.path.join(feature_extraction_directory, 'pre_early_late_contrast_phases_T0.csv'), index_col=0)
    time_information.index = time_information.index.astype('str')
    mask_name = 'new_FTV_GLCM_SS_T0_mask.mha'
    radiomics_feature_extraction(database_directory, time_information, mask_name, feature_extraction_directory)
    radiomics_feature_extraction_with_normalization(database_directory, time_information, mask_name, feature_extraction_directory)

    # compare the feature values between with and without image normalization for the three demo patients. The results should be zeros for all the three patients
    feature_name = 'log_sigma_3_mm_3D_glcm_SumSquares'
    normalization_comparison(feature_name, feature_extraction_directory)

