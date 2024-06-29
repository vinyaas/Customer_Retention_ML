import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

#Finding the unique values of each feature for encoding.
def display_unique(df):
    for i in df.columns:
        print(f"Column name : {i}")
        print(f"Unique Values : {df[i].unique()} \n")

## LABEL ENCODING  
from sklearn.preprocessing import LabelEncoder

class LabelEncoding:
    
    def label_encoder(df):
        le = LabelEncoder()
        for i in df.columns:
            df[i] = le.fit_transform(df[i])
        return df
    
label = LabelEncoding
label.label_encoder(df)

## FEATURE SELECTION - PEARSON CO-EFFICIENT
#Always do feature selection on training data to avoid data leakage to test data .

class pearson:
    
    def __init__(self , max_features):
        self.max_features = max_features

    def pearsons_corr(self , x_train , y_train):
        
        df_train = pd.DataFrame(x_train)
        df_train['Churn'] = y_train
        
        correlations = df_train.corr()['Churn'].drop('Churn')
        sorted_corr = abs(correlations).sort_values(ascending= False)[:self.max_features]
        selected_features = sorted_corr.index[:self.max_features].tolist
        return selected_features , sorted_corr

x = df.drop('Churn' , axis = 1)
y = df['Churn']       
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.25 , random_state= 42)

corr = pearson(max_features=8)
selected_features , sorted_corr = corr.pearsons_corr(x_train= x_train , y_train= y_train)
print(f"{selected_features}\n\n{sorted_corr}")
#[['Contract', 'tenure_group', 'OnlineSecurity', 'TechSupport','TotalCharges', 'OnlineBackup', 'PaperlessBilling', 'MonthlyCharges' , 'Churn']]

## FEATURE SELECTION - FORWARD FEATURE SELECTION
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class FeatureSelector:
    
    def __init__(self , max_features):
        self.max_features = max_features
        
    def forward_selection(self , x_train , y_train , x_test , y_test):
        
         #initializing the lists for storing the selected features and scores 
        selected_features = []
        ordered_scores = []
        
        #iterate over the range of maximum features
        for i in range(self.max_features):
            features_left = list(set(x_train.columns) - set(selected_features))
            
            #initialize the best score to 0 and best feature to empty string 
            best_score = 0
            best_feature = ''
            
            #iterate over each feature left to be selected 
            for feature in features_left:
                #create a temporary list of features and append the current feature to it 
                temp_selected_features = selected_features + [feature]
                
                #train a decision tree classifier on the training data with temporary selected features
                classifier = DecisionTreeClassifier().fit(x_train[temp_selected_features], y_train)
                
                #predict the test data and calculcate the accuracy 
                y_pred = classifier.predict(x_test[temp_selected_features])
                score = accuracy_score(y_test , y_pred)
                
                #if current score is better than best score update the best score and best feature
                if score > best_score:
                    best_score = score 
                    best_feature = feature
                    
            #append the best feature and its score to respective lists 
            selected_features.append(best_feature)
            ordered_scores.append(best_score)
            print(f"Feature {i+1} : {best_feature} with score {best_score}\n")
            
        return selected_features , ordered_scores
    
x = df.drop('Churn' , axis = 1)
y = df['Churn']       
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.25 , random_state= 42)

fs = FeatureSelector(max_features= 9 )
selected_features , ordered_scores = fs.forward_selection(x_train , y_train , x_test , y_test)

print(f"Selected Features : {selected_features} \n\nOrdered Scores : {ordered_scores}")
# [['DeviceProtection', 'tenure_group', 'InternetService', 'PaymentMethod', 'PhoneService', 'PaperlessBilling', 'TechSupport' , 'Churn']]

## DIMENSIONALITY REDUCTION - PRINCIPAL COMPONENT ANALYSIS

class PrincipalComponentAnalysis:
    
    def __init__(self, target_column, n_components ):
        self.target_column = target_column
        self.n_components = n_components

    def pca(self , df):
        df_pca = df.copy().drop(self.target_column , axis = 1)

        #standardization of the data 
        scaler = StandardScaler()
        scaler.fit(df_pca)
        scaled_data = scaler.transform(df_pca)

        #Applying PCA Algorithm
        pca = SKPCA(n_components=self.n_components)
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data)
        pca_df = pd.DataFrame(data = x_pca, columns = ['pca_1', 'pca_2', 'pca_3'])
        #Adding pca columns and data to the dataset df
        df = pd.concat([df,pca_df] , axis = 1)
        return df
    
dr = PrincipalComponentAnalysis( 'Churn', 3)
df = dr.pca(df)

#exporting the data 
df.to_pickle("../../data/processed/01_Feature_Engineered_data.pkl")

        

        
        
        



            
    





