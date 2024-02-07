
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

if __name__ == '__main__':
    database_directory = 'demo_image_dataset'
    feature_extraction_directory = 'feature_extraction'
    time_information = pd.read_csv(os.path.join(feature_extraction_directory, 'pre_early_late_contrast_phases_T0.csv'), index_col=0)
    time_information.index = time_information.index.astype('str')
    mask_name = 'new_FTV_GLCM_SS_T0_mask.mha'
    radiomics_feature_extraction(database_directory, time_information, mask_name, feature_extraction_directory)

