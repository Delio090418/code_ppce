import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


df = pd.read_csv('/Users/delio/Documents/Working projects/Balazs/code_ppce/healthcare-dataset-stroke-data.csv')


df.drop(['id'],axis=1,inplace=True)

#print(df.shape)
# 

# df = df.dropna()


#count of missing data
missing_values_count = df.isna().sum()

#find the percentage of missing data
total_cells = np.prod(df.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing / total_cells) * 100
# print("Percentage of missing data from the dataset is : {}%".format(percent_missing))

df['bmi']=df['bmi'].fillna(df['bmi'].mean())
#print(df.head())
#print(df.info)

df.drop(df[df['gender'] == 'Other'].index, inplace = True)
#print(df["gender"].value_counts())

# print("The number of people who don't have stroke : ", df['stroke'].value_counts()[0])
# print("The number of people who don't have stroke : ", df['stroke'].value_counts()[1])
cond1 = df['avg_glucose_level'] > 170
cond2 = df['stroke'] == 1
# print("The number of outliers in avg_glucose_level with stroke = 1 are : ", df[cond1 & cond2].shape)
cond3 = df['bmi'] > 47
cond4 = df['stroke'] == 1
# print("The number of outliers in bmi with stroke = 1 are : ", df[cond3 & cond4].shape)

# print("The shape before removing the BMI outliers : ",df.shape)
df.drop(df[df['bmi'] > 47].index, inplace = True)
# print("The shape after removing the BMI outliers : ",df.shape)

# # print(df.dtypes)

object_cols = ["gender","ever_married","work_type","Residence_type","smoking_status"]
label_encoder = LabelEncoder()
for col in object_cols:
    label_encoder.fit(df[col])
    df[col] = label_encoder.transform(df[col])


sampler = SMOTE(random_state = 42)
X_data= df.drop(['stroke'],axis=1)
y_labels= df[['stroke']]
X_data,y_labels= sampler.fit_resample(X_data,y_labels['stroke'].values.ravel())
y_labels= pd.DataFrame({'stroke':y_labels})
#sns.countplot(data = y, x = 'stroke', y= None)
#plt.show()

# Joining back dataset
df = pd.concat([X_data,y_labels],axis = 1)
# print(df.head())
#print(X_data.shape)


# idx_0 = np.where(y_labels == 0)[0]
# idx_1 = np.where(y_labels== 1)[0]

# print(idx_0.shape)
# print(idx_1.shape)