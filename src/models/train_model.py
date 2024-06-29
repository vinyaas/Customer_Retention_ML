#importing neccessary libraries
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from imblearn.combine import SMOTEENN
import pickle

df = pd.read_pickle("../../data/processed/01_Feature_Engineered_data.pkl")

## Extracting Feature subsets from feature engineering ------------------------------------------
basic_features = df.drop('Churn' , axis = 1).columns.tolist()
Pearson_features = ['Contract', 'tenure_group', 'OnlineSecurity', 'TechSupport','TotalCharges', 
                    'OnlineBackup', 'PaperlessBilling', 'MonthlyCharges']
pca_features = ['pca_1' , 'pca_2' , 'pca_3']
fs_features = ['DeviceProtection', 'tenure_group', 'InternetService', 'PaymentMethod', 'PhoneService', 'PaperlessBilling', 'TechSupport']

#combining all the subsets 
feature_sets = [basic_features, Pearson_features , pca_features , fs_features]
target = df['Churn']
df[['DeviceProtection', 'tenure_group', 'InternetService', 'PaymentMethod', 'PhoneService', 'PaperlessBilling', 'TechSupport']].head(1)

## ------------------------------------DECISION TREE MODEL --------------------------------------------
class Decision_tree_model:
    def __init__(self, df, target, feature_sets):
        self.df = df
        self.target = target
        self.feature_sets = feature_sets
        self.x_train = None
        self.y_train = None

    def train_model(self):
        for i, features in enumerate(self.feature_sets):
            print(f"Training model with feature set {i + 1} ...")

            self.x_train, x_test, self.y_train, y_test = train_test_split(self.df[features], self.target, test_size=0.25, random_state=42)
            dtree = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=4, random_state=100)
            dtree.fit(self.x_train, self.y_train)

            y_pred_train = dtree.predict(self.x_train)
            y_pred_test = dtree.predict(x_test)    

            accuracy_train = metrics.accuracy_score(self.y_train, y_pred_train)
            accuracy_test = metrics.accuracy_score(y_test, y_pred_test)

            cp_train = metrics.classification_report(self.y_train, y_pred_train, target_names=["Not Churned", "Churned"])
            cp_test = metrics.classification_report(y_test, y_pred_test, target_names=["Not Churned", "Churned"])

            print(f"Train Accuracy with feature set {i+1}: {accuracy_train}")
            print(f"Test Accuracy with feature set {i+1}: {accuracy_test}\\n")
            print(f"Classification report for train data with feature set {i + 1} : {cp_train}")
            print(f"Classification report for test data with feature set {i + 1} : {cp_test}")
    
    #Hyper-parameter tuning    
    def perform_grid_search(self):
        
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'min_samples_leaf': [1, 2, 4, 8, 16]
        }
        dbtree = DecisionTreeClassifier()
        grid_search = GridSearchCV(estimator=dbtree, param_grid=param_grid, cv=10, scoring='accuracy') 
        grid_search.fit(self.x_train, self.y_train)
        return grid_search.best_params_
            
dt = Decision_tree_model(df, target, feature_sets)
dt.train_model()
dt.perform_grid_search()

# Here we can see Feature set 4 provides highest accuracy among these sets 
# We have performed forward feature selection using a simple decision tree for this feature set.
# Feature Set 4 : 

    # Train Accuracy: 0.799
    # Test Accuracy: 0.802
    # Precision (Train): Not Churned - 0.83, Churned - 0.66
    # Precision (Test): Not Churned - 0.83, Churned - 0.68
    # Recall (Train): Not Churned - 0.91, Churned - 0.50
    # Recall (Test): Not Churned - 0.91, Churned - 0.52
    
## BUT WE CAN SEE THERE IS A HIGH NUMBER OF IMBALANCED CLASSES SO WE WILL BE USING THE SMOTE-ENN TECHNIQUE .
# WE WILL BE CONSIDERING FEATURE SET 4 AS IT GAVE HIGHER ACCURACY AND TRY IMPROVING IT BY RESAMPLING THE DATASET .

def SMOTTEN(fs_features):
    
    print(f"Training model for fs_features subeset : \n")
    x_sm = df[fs_features]
    y_sm = df['Churn']
    sm = SMOTEENN()
    
    x_resampled , y_resampled  = sm.fit_resample(x_sm , y_sm)
    xr_train , xr_test , yr_train , yr_test = train_test_split(x_resampled , y_resampled , test_size= 0.25 , random_state= 102)
    dt_smote = DecisionTreeClassifier(criterion='gini' , max_depth=6 , min_samples_leaf=4 , random_state= 101)
    dt_smote.fit(xr_train , yr_train)
    ypred_smote = dt_smote.predict(xr_test)
    accuracy_sm = metrics.accuracy_score(ypred_smote , yr_test)
    
    print(f"Accuracy of fs_features : {accuracy_sm}\n")
    print(metrics.classification_report(yr_test , ypred_smote , labels= [0 ,1]))
    return dt_smote , xr_test , yr_test
 
dt_smote_model , xr_test_model , yr_test_model = SMOTTEN(fs_features)

#--> Accuracy of fs_features : 0.9845454545454545

# Classification Report :
#                 precision    recall  f1-score   support

#           0       0.99      0.98      0.99       738
#           1       0.96      0.99      0.98       362

#    accuracy                           0.98      1100
#   macro avg       0.98      0.99      0.98      1100
#weighted avg       0.98      0.98      0.98      1100

#Importing the model as a PICKLE File .

filename = 'model.sav'
pickle.dump(dt_smote_model, open(filename , 'wb'))

load_model = pickle.load(open(filename , 'rb'))
load_model.score(xr_test_model , yr_test_model)

#We are getting 98 percent accuracy on the pickle file also .













