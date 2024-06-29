import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#reading the dataset 

df = pd.set_option('display.max_columns' , None)
df = pd.read_csv('../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')


#calculating the percentage of target value distributiion
plt.title("Target variable distribution")
df['Churn'].value_counts().plot(kind = 'barh' , color = 'green' , edgecolor = 'black')

(df['Churn'].value_counts() / len(df)) *100

#-----------------------    DATA CLEANING -----------------------------

#check for missing values

def missing_values(df):
    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (total / len(df)) * 100
    return pd.concat((total , percent) , axis = 1 , keys = ['TOTAL' , 'PERCENT'])

miss_values = missing_values(df)
miss_values

#checking the datatype of the columns

df.info()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'] , errors = 'coerce') #coerce converts values which cannot be converted to numbers to Nan

#lets check missing values now 
miss_values

#dropping the na values 
df.dropna(inplace = False)

#Grouping the tenure to groups of 12 months 

def range_of_tenure(df):
    return df['tenure'].min()  , df['tenure'].max()
range_of_tenure(df)

bins = [0 , 12 , 24 , 36 , 48 , 60 , 72]
labels = ['0-12' , '13-24' , '25-36' , '37-48' , '49-60' , '61-72']

df['tenure_group'] = pd.cut(df['tenure'] , bins = bins , labels = labels , include_lowest= True)

#removing unecessary columns

df.drop(columns = ['customerID' , 'tenure'] , axis = 1, inplace= True)

#Display all the columns with its unique values

def displayunique(df):
    for column in df.columns:
        print(f"Column Name : {column}")
        print(f"Unique Values : {df[column].unique()}\n")
        
displayunique(df)

#Creating a function with all the above process 

def make_dataset(df):
    
    #reading the dataset
    df = pd.set_option('display.max_columns' , None)
    df = pd.read_csv('../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    #fixing the datatype of the column 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'] , errors = 'coerce')
    
    #dropping the Nan Values 
    df.dropna(inplace = False)
    
    #grouping the tenure group as a group of 12 months
    bins = [0 , 12 , 24 , 36 , 48 , 60 , 72]
    labels = ['0-12' , '13-24' , '25-36' , '37-48' , '49-60' , '61-72']
    df['tenure_group'] = pd.cut(df['tenure'] , bins = bins , labels = labels , include_lowest= True)
    
    #dropping the uncessary columns
    df.drop(columns = ['customerID' , 'tenure'] , axis = 1, inplace= True)
    
    return df

make_dataset(df)

#export the cleaned dataset 

data_processed = df
data_processed.to_pickle("../../data/interim/01_data_processed.pkl")
