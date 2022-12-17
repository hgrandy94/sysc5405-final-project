import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import sklearn
import glob
import imblearn
from sklearn import metrics
from imblearn.over_sampling import ADASYN 
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from scipy import stats




# Import feature data

Data = pd.read_csv (r'D:\Documents\Grad School\Coursework\BIOM5405\Project\TrainData\Features_All.csv') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'

feature_names = Data.columns.values.tolist()

#Test Train Split: 70/30
Train, Test = sklearn.model_selection.train_test_split(Data, test_size=0.3, train_size=0.7, random_state=None, shuffle=True, stratify=None)


# Normalize Training Data
Train_norm = Train.copy()
Test_norm = Test.copy()

Norm = StandardScaler() #Using standard scaler to normalize the data
Norm.fit(Train_norm.iloc[:,2:1026]) #Fit standard scaler to training data
Train_norm.iloc[:,2:1026] = Norm.transform(Train_norm.iloc[:,2:1026]) #Transform training data to normalized scalar
Test_norm.iloc[:,2:1026] = Norm.transform(Test.iloc[:,2:1026]) #Normalize test data



## Outlier removal function (Waiting on confirmation)
# Create function that computes mean, cov matrix, and inv cov matrix
# X_train is the training dataset
def get_mean_cov(TrainingData):
    # Merge dfs
  #  norm_x_df = pd.DataFrame(X_train, columns=feature_names)
  #  norm_df = pd.concat([y_train, norm_x_df], axis=1)
    norm_df = TrainingData 
  # Compute mean and cov per class per feature
    avg_list = []
    cov_list = []
    inv_cov_list = []
    for i in range(5):
        # Compute mean
        avg = np.mean(norm_df[norm_df["class"]==i][feature_names[2:-1]], axis=0)
        avg_list.append(avg)
        # Compute cov matrix
        cov = np.cov(norm_df[norm_df["class"]==i][feature_names[2:-1]], rowvar=False)
        cov_list.append(cov)
        # Compute inverse of cov matrix
        inv_cov = np.linalg.inv(cov)
        inv_cov_list.append(inv_cov)
    return norm_df, avg_list, inv_cov_list
# Test get_mean_cov function
norm_df, avg_list, inv_cov_list = get_mean_cov(Train_norm)
# Determine which features should be removed (identify outliers based on Mahalanobis dist)

# Create function that computes Mahalanobis distance and adds it to norm_df
def get_mahalanobis_dist(label, features):
    u = avg_list[label]
    v = features
    vi = inv_cov_list[label]
    delta = u - v
    m = np.dot(np.dot(delta, vi), delta)
    #dist = distance.mahalanobis(u, features, vi)
    return np.sqrt(np.abs(m))

# Call function for each feature
norm_df["mahalanobis_dist"] = norm_df.apply(lambda row: get_mahalanobis_dist(int(row["class"]), row[feature_names[2:-1]]), axis=1)
norm_df["mahalanobis_dist"].describe()
# Initialize list for distances
# dist = np.zeros(norm_df.shape[0])
# for i, row in norm_df.iterrows():
#     dist[i] = get_mahalanobis_dist(int(row["class"]), row[feature_names])
# Drop outliers
def drop_outliers(norm_df, threshold):
    thresh = threshold
    norm_df.sort_values(by="mahalanobis_dist", ascending=False, inplace=True)
    norm_df.reset_index(inplace=True, drop=True)
    norm_df.drop(norm_df.index[:int(norm_df.shape[0]*thresh)], inplace=True)
    norm_df.reset_index(inplace=True, drop=True)
    return norm_df
# Test drop_outliers function
norm_df = drop_outliers(norm_df, 0.1)
# Print updated descriptive stats
norm_df["mahalanobis_dist"].describe()
#print(len(norm_df["class"])) (edited) 



#Balance Classes With ADASYN

X = norm_df.iloc[:,2:1026]
y = norm_df['class']

ada = ADASYN(sampling_strategy='all', random_state=(255), n_neighbors=4)

X_resampled, y_resampled = ada.fit_resample(X, y)

#Check new class counts 
y_resampled.value_counts()


#Fit Initial SVM to Balanced Data
from sklearn.svm import SVC
SVM = SVC(C=1, kernel = 'linear', gamma = 1, probability = True)
SVM.fit(X_resampled,y_resampled)
SVM_pred = SVM.predict(Test_norm.iloc[:,2:1026])

#Get initial/Baseline F1 Score
f1_acc_ADASYN = metrics.f1_score(Test_norm['class'], SVM_pred, average = 'macro')



## Recursive Feature Elimination
nfeat = 1
f1_acc_RFE = pd.DataFrame(np.zeros([100, 2]))
k=0

while (nfeat>0):
    selector = RFE(estimator = SVM, n_features_to_select= nfeat, step=2)
    selector = selector.fit(X_resampled, y_resampled)
    features = selector.support_ #Gives True/False values for each feature
#Initialize dataframes for features selected by RFE
    Train_RFE = pd.DataFrame(np.zeros([len(X_resampled.index), np.count_nonzero(features)]))
    Test_RFE = pd.DataFrame(np.zeros([len(Test), np.count_nonzero(features)]))
# Reset indexing 
    Test_ = Test_norm.sort_index(ascending=True)
    Test_ = Test_.reset_index(drop=True) #need to do this to deal with initial indexing

# Loop: if feature determined important, populate new training and test data using that feature
    i=0
    j=0
    for i in range(len(features)):
        if (features[i] == True):
            Train_RFE.iloc [:,j]= X_resampled.iloc[:,i]
            Test_RFE.iloc [:,j]= Test_.iloc[:,(i+2)]
            j=j+1
            i = i+1
        else:
            i=i+1        
#Retrain SVM based only on RFE Features
    SVM_RFE = SVC(C=1, kernel = 'linear', gamma = 1, probability = True)
    SVM_RFE.fit(Train_RFE,y_resampled)
    SVM_RFE_pred = SVM_RFE.predict(Test_RFE)
#Get F1 Score for SVM with RFE features
    f1_acc_RFE.iloc[k,0] = metrics.f1_score(Test_['class'], SVM_RFE_pred, average = 'macro')    
    f1_acc_RFE.iloc[k,1] = nfeat
    k=k+1
    nfeat=nfeat-0.01




## RFECV

#Reset counter and initialize dataframe for F1 scores
nfeat = 1
f1_acc_RFECV = pd.DataFrame(np.zeros([100, 2]))
k=0

while (nfeat>0):
    selector_CV = RFECV(estimator = SVM, min_features_to_select= nfeat, step=2, cv=5)
    selector_CV = selector_CV.fit(X_resampled, y_resampled)
    features_CV = selector_CV.support_ #Gives True/False values for each feature
#Initialize dataframe for new train and test data
    Train_RFECV = pd.DataFrame(np.zeros([len(X_resampled.index), np.count_nonzero(features_CV)]))
    Test_RFECV = pd.DataFrame(np.zeros([len(Test), np.count_nonzero(features_CV)]))
# Reset indexing
    Test_ = Test_norm.sort_index(ascending=True)
    Test_ = Test_.reset_index(drop=True) #need to do this to deal with initial indexing

    i=0
    j=0
    for i in range(len(features)):
        if (features_CV[i] == True):
            Train_RFECV.iloc [:,j]= X_resampled.iloc[:,i]
            Test_RFECV.iloc [:,j]= Test_.iloc[:,(i+2)]
            j=j+1
            i = i+1
        else:
            i=i+1        
#Retrain SVM based only on RFECV Features
    SVM_RFECV = SVC(C=1, kernel = 'linear', gamma = 1, probability = True)
    SVM_RFECV.fit(Train_RFECV,y_resampled)
    SVM_RFECV_pred = SVM_RFECV.predict(Test_RFECV)
#Get F1 Score for SVM with RFECV features
    f1_acc_RFECV.iloc[k,0] = metrics.f1_score(Test_['class'], SVM_RFECV_pred, average = 'macro')    
    f1_acc_RFECV.iloc[k,1] = nfeat
    k=k+1
    nfeat=nfeat-0.01



## Plot F1 scores against Number of Features
plot.plot (f1_acc_RFE[1], f1_acc_RFE[0], label='RFE', color = 'indianred')
plot.plot (f1_acc_RFECV[1], f1_acc_RFECV[0], label='RFECV', color='steelblue')
plot.axhline (f1_acc_ADASYN, label='No Feature Elimination', color = 'black')
plot.xlabel('Percentage of Features')
plot.ylabel('F1 Macro Score')
plot.title('Recursive Feature Elimination')
plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))



# Make array with common features from RFE and RFECV
common_features = np.logical_and(features, features_CV)
common_features = np.nonzero (common_features)
common_features = np.asarray (common_features)

# Import Azure Features
Azure = pd.read_csv (r'D:\Documents\Grad School\Coursework\BIOM5405\Project\TrainData\topfeatures-automl.csv') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
Azure = Azure.to_numpy()

# Make boolean array similar to features and features_CV
Azure_bool = np.zeros((1024), dtype=bool)

i = 0

for i in range (len(Azure)):
    Azure_bool[Azure[i]] = True
    i = i+1
    
# Make array with common features from RFE and RFECV and Azure
common_features_Az = np.logical_and(Azure_bool, features_CV)
common_features_Az = np.nonzero (common_features_Az)
common_features_Az = np.asarray (common_features_Az)

np.savetxt("CommonFeatures_RFE_RFECV.csv", common_features, delimiter=",")
np.savetxt("CommonFeatures_RFE_RFECV_Azure.csv", common_features_Az, delimiter=",")
