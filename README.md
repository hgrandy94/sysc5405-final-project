# SYSC 5405 Final Project
## Group 9 - Support Vector Machines

This repository contains the contents of the SYSC 5405 final project (fall 2022 semester).

Each artifact in the repository is described below. Note that it is not possible to run any of the Jupyter Notebooks or Python code with this repository alone as the image feature vectors were not included in the repository.

### Code Folder

- *data_exploration.ipynb* [Jupyter Notebook]

  This notebook contains Python code from the initial data exploration described in section 2.1. This includes creating the single CSV file   with all image feature vectors, investigating class prevalence, and testing Linear Discriminant Analysis for feature reduction as well as K-means clustering to observe natural clusters in the dataset.

- *model_training.ipynb* [Jupyter Notebook]

  This notebook contains the model training pipeline code developed for both the final SVM model, and the meta-learning classifier. This notebook contains the core logic of the model development. Each step in the pipeline is clearly labeled using a markdown cell to enhance readability of the notebook.
  
- *model_inferencing.ipynb* [Jupyter Notebook]

  This notebook contains the model inferencing pipeline code developed to generate predictions from the SVM model. 

- *rfe_feature_elimination.ipynb* [Jupyter Notebook]

  This Python script contains many of the data preprocessing pipeline steps included in the model_training notebook. The primary purpose of this script is to perform recursive feature elimination and output the results to a CSV file to be reloaded into the model_training pipeline notebook.
  
- *environment.yml* [YAML file]

  This file describes the Python environment required to execute the Jupyter Notebooks and Python script. Conda was used as the environment management tool for this project, but the YAML file can be imported into most Python environment management tools.

- *Group9-ProjectTesting.ipynb* [Jupyter Notebook]

  This notebook is stored in another GitHub repository, which can be found here: https://github.com/Junebuggi/Group9_SYSC5405/blob/main/Group9-ProjectTesting.ipynb. It contains additional data exploration code, including the histograms used to visually inspect the feature data.
 
### Data Folder

- *f1_estimates/training_pipeline_results.csv* [CSV file]

  This CSV file contains the F1 macro average estimates obtained using different random seeds for both the single SVM model and the meta-learning classifier. The columns in this CSV are: F1Score (F1 macro average from the SVM model), EnsembleF1Score (F1 macro average from the voting classifier), and RandomSeed (the random seed used in that iteration).
  
- *feature_selection/CommonFeatures_RFE_RFECV.csv* [CSV file]

  This CSV file contains the common features identified from recursive feature elimination and recursive feature elimination with 5-fold cross-validation.

- *feature_selection/azureml_designer_features.csv* [CSV file]

  This CSV file contains the top 100 features determined using Pearson correlation from the Azure Machine Learning Designer Filter-Based Feature Selection module.

- *predictions/group_09.csv* [CSV file]

  This CSV file contains the final predictions on blind test data that the group submitted for the class competition. The columns in this CSV are: uid (unique identifier of the sample image), and class (predicted class).

- *training_data/full_dataset.csv* [CSV file]

  This CSV file contains unique identifiers, class, and 1024 image feature vectors for all samples in the training dataset.
 
- *training_data/train.csv* [CSV file]

  This CSV file contains the unique identifier and class for all samples in the training dataset. This file was provided at the beginning of the project and, along with individual CSV files containing image feature vectors for each sample, was used to construct the full_dataset CSV file.

### Plots Folder
- The plots folder contains 1024 boxplots - one boxplot for feature - in PNG format. 
