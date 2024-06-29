import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn  as sns

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

#displaying all the unique values 

def displayunique(df):
    for column in df.columns:
        print(f"Column Name : {column}")
        print(f"Unique Values : {df[column].unique()} \n")
        
#visualize the data distribution
def pie_plots(df):
    fig , axs = plt.subplots( 3 , 2 , figsize = (15 , 15))
    axs = axs.flatten()
    for i in range(6):
        column = df.columns[i]
        values = df[column].value_counts()
        axs[i].pie(values , labels = values.index , autopct = "%1.1f%%" , startangle = 90)
        axs[i].set_title(column)
        
    axs[5].axis('off')
    fig.suptitle("Percentage of Data Distribution")
    plt.tight_layout()
    plt.show()
    
#Plot distribution by individual predictors based on churn

independent_variables = df.drop(columns = ['MonthlyCharges' , 'TotalCharges' , 'Churn'])
for i in independent_variables:
    plt.figure(figsize = (12 , 6))
    plt.title( "Count plot of " + i)
    sns.countplot(data = df , x = i , hue= "Churn")
    plt.savefig(f"../../reports/figures/{i.title()}.png")
    plt.show()
    

    


    

