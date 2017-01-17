------------------------------------------------------------
TIME SERIES CLASSIFICATION USING FEATURE COVARIANCE MATRICES
------------------------------------------------------------

Hamza Ergezer, Kemal Leblebicioglu
January 2017

The code repository mainly contains the following files:

- TSC_CovFeatures_Batch43.m: Batch run of the proposed method for 43 datasets. It returns also calculation time of the method.
- getDatasetNamesUCR43.m: List of 43 datasets.
- demo_TSC_CovFeatures: The code executes the proposed method for "Adiac" dataset. It can be used to test other dataset. 
- crossValidation.m: The code determines the optimum hyperparameters for "Adiac" dataset. It may take so much time due to random selection of training and testing samples during cross-validation.
- optParams.mat: Optimum parameters obtained by cross-validation 


To run the code, the following steps must be taken.

1. Download the UCR Datasets from the website http://www.cs.ucr.edu/~eamonn/time_series_data/UCR_TS_Archive_2015.zip
2. UCR datasets must be in a folder "UCR_Data" in your working directory. Folder notations should be like ".\UCR_Data\Adiac\"

The code is only for the academic purposes. 