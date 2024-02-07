import pandas as pd
import os
import numpy as np



#
#
def feature_binarization(feature_directory):
    image_feature_table = pd.read_csv(os.path.join(feature_directory, 'image_features.csv'), index_col=0)
    binarized_feature_table = {}
    for feature_name in image_feature_table.columns:
        binarized_feature = (image_feature_table[feature_name] >= image_feature_table[feature_name].median()).astype(int)
        binarized_feature_table[feature_name] = binarized_feature
    binarized_feature_table = pd.concat(binarized_feature_table, axis=1)
    binarized_feature_table.to_csv(os.path.join(feature_directory, 'image_features_binarized.csv'))



if __name__ == '__main__':
    feature_binarization('features')




