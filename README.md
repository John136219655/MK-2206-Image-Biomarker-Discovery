# Overview
This repositorty aims to provide the Python source code, demo image data, and complete features tables for the reproduction of the manuscript results. Each python file is responsible for a specific step in data analysis.

# System requirements
The results were generated by a PC equipped with Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz and 32 GB RAM on the Windows 10 Pro (22H2) operating system. There is no required non-standard hardware required.
## List of python packages required:
* PyRadiomics (3.0.1)
* Numpy (1.23.0)
* Pandas (2.2.0)
* Scikit-learn (1.0.2)
* Scipy (1.8.0)
* Statsmodels (0.13.5)
* Pymc (5.0.2)
* Arviz (0.14.0)
* Pytensor (2.9.1)
* Matplotlib (3.7.4)
* Seaborn (0.12.0)

# Installation guide
No special installtion is required to run the individual python files. Users are recommended to setup a new python (3.11.0) environment for reproducing the study results.

# Demo
## Feature extraction
A demo image dataset was provided in the "demo_image_dataset" folder. All the images and segmentation files of one patient were organized by one folder named after the patient ID. 
Each DCE-MRI image was named by the convention "DCE_MRIT0_Phase_n.mha" with n starting from 0 as the phase number. The FTV segmentation file has the name of "new_FTV_GLCM_SS_T0_mask.mha".
Users can perform feature extraction by running the Python file "feature_extraction.py". The extracted feature values will be exported as a csv table named "image_features.csv" in "feature_extraction" folder.
The complete radiomics feature table can be found in the "features" folder. The expted run time for the three-patient demo dataset is 5 minutes. The actual run time may vary depending on the hardware.

This module also contains a small experiment that compare the selected imaging biomarker "log_sigma_3_mm_3D_glcm_SumSquares" between the extraction with and without image normalization.
Another set of features were extracted based on a new parameter file "feature_extraction_parameters_with_normalization.yaml" that contains the settings for image normalization. A new feature table is generated and exported as "image_features_with_normalization.csv".
Finally, the results show that the feature values are identical with and without image normalization.
# Instructions for use
## Feature processing
All the radiomics features were binarized by the median value. Users can run the "feature_processing.py" file to perform the feature binarization. 
A binarized feature table named "image_features_binarized.csv" will be exported to the "features" folder.
## Biomarker selection
Users can perform image biomarker selection by running the python file "image_biomarker_discovery.py". 
The "clinical_features.csv", "image_features.csv", and "iccs.csv" will be used as the input data for biomarker selection.
The selection results will be exported to the "biomarker discovery folder" where the selected features and their statistical analysis results can be found in the "selected_biomarkers.csv" file.
## Biomarker assessment
Users can analyze the clinical values of the selected biomarker by running the python file "biomarker_assessment.py". 
It performes both the continuous biomarker performance and binarized biomarker performance using the radiomics feature tables "image_features.csv" and "image_features_binarized.csv". 
A complete bayesian analysis pipeline for all the pre-defined subtypes is also included. All the assessment results are exported to the "biomarker_assessment" folder. 
Continues performance can be found in the "univariate_performance.csv" file, and all the bayesian posterior distributions can be found in the "GLCM_SS\bayesian" subfolder.
## Statistical analysis
Users can perform statistical analysis by running the python file "statistical_analysis.py'. All the outputs of statistical analysis are located in the "statistical_analysis" folder.
Statistical comparisons of baseline characteristics are exported as "patient_characteristics_comparison_treatment.csv" file. Comparisons of estimated pCR rates and pCR rate gains are exported as "GLCM_SS+_bayesian_comparison_stats.csv".


