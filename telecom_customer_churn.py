#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# In[3]:


# Load the dataset into a Pandas dataframe using read_csv
df = pd.read_csv('telecom_customer_churn.csv')
print(df)


# In[ ]:





# In[4]:


# View the first 5 rows of the dataframe using head
print(df.head(5))


# In[5]:


# View the last 5 rows of the dataframe using tail
print(df.tail(5))


# In[17]:


# View the shape of the dataframe using shape
print(df.shape)


# In[18]:


# View the column names of the dataframe using columns
print(df.columns)


# In[19]:


# View the datatypes of each column using dtypes
print(df.dtypes)


# In[20]:


# Check for missing values in the dataframe using isna
print(df.isna())


# In[21]:


# Count the number of missing values in each column using isna and sum
print(df.isna().sum())


# In[6]:


# Remove rows with missing values using dropna
df = df.dropna()
df


# In[25]:


# Fill missing values with a specified value using fillna
df = df.fillna(0)
df


# In[24]:


# Drop duplicate rows using drop_duplicates
df = df.drop_duplicates()
df


# In[7]:


# Rename a column using rename
df = df.rename(columns={'gender': 'Gender'})
df


# In[ ]:





# In[5]:


# Change the datatype of a column using astype
#df['TotalCharges'] = df['TotalCharges'].astype(float)
#df['TotalCharges'] = df['TotalCharges'].replace('', np.nan).astype(float)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').astype(float)


# In[8]:


# Select multiple columns of the dataframe using indexing
print(df[['Gender', 'Partner', 'Dependents']])


# In[9]:


# Select rows based on a condition using boolean indexing
print(df[df['MonthlyCharges'] > 100])


# In[10]:


# Sort the dataframe by a column using sort_values
df.sort_values('TotalCharges', ascending=False)


# In[12]:


# Filter rows based on a condition using query
df.query('tenure > 10 and MonthlyCharges < 50')


# In[13]:


# Group the dataframe by a column using groupby
grouped = df.groupby('PaymentMethod')
grouped


# In[14]:


# Aggregate data using agg
print(grouped.agg({'MonthlyCharges': 'mean', 'TotalCharges': 'sum'}))


# In[15]:


# Merge two dataframes using merge
df1 = pd.DataFrame({'customerID': [1, 2, 3], 'gender': ['Male', 'Female', 'Male']})
df2 = pd.DataFrame({'customerID': [2, 3, 4], 'age': [25, 30, 35]})
merged = pd.merge(df1, df2, on='customerID')
merged


# In[16]:


# Join two dataframes using join
df1 = pd.DataFrame({'customerID': [1, 2, 3], 'gender': ['Male', 'Female', 'Male']})
df2 = pd.DataFrame({'age': [25, 30, 35]}, index=[1, 2, 3])
joined = df1.join(df2, on='customerID')
joined


# In[17]:


# Create a new column based on existing data using apply
exitingdata= df.loc[:, 'TenureMonths'] = df['tenure'] * 12
exitingdata


# In[20]:


# Create a new column based on multiple columns using apply and lambda functions
#df['TotalChargesDiscount'] = df.apply(lambda x: x['TotalCharges'] * 0.1 if x['MonthlyCharges'] > 100 else 0, axis=1)
# Convert 'TotalCharges' column to numeric type
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Create a new column based on multiple columns using apply and lambda functions
df['TotalChargesDiscount'] = df.apply(lambda x: x['TotalCharges'] * 0.1 if x['MonthlyCharges'] > 100 else 0, axis=1)


# In[21]:


# Create a new column using np.where')
#df.loc[:, 'HighTenure'] = np.where(df['tenure'] > 60, 'Yes', 'No')
import warnings
warnings.filterwarnings("ignore")
highnew = df.loc[:, 'HighTenure'] = np.where(df['tenure'] > 60, 'Yes', 'No').copy()
highnew


# In[22]:


# Convert a column to datetime format using to_datetime
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'Contract': ['Month-to-month', 'One year', 'Two year']
})

# Create a new column for ContractStartDate
ContractStartDate= df['ContractStartDate'] = pd.NaT

# Convert a column to datetime format using to_timedelta
df.loc[df['Contract'] == 'One year', 'ContractStartDate'] = pd.Timestamp.now() - pd.to_timedelta('365D')
df.loc[df['Contract'] == 'Two year', 'ContractStartDate'] = pd.Timestamp.now() - pd.to_timedelta('730D')

# Alternatively, you can use pandas DateOffset to subtract years:
# df.loc[df['Contract'] == 'One year', 'ContractStartDate'] = pd.Timestamp.now() - pd.DateOffset(years=1)
# df.loc[df['Contract'] == 'Two year', 'ContractStartDate'] = pd.Timestamp.now() - pd.DateOffset(years=2)

# Convert the Month-to-month contracts to the end of the current month
df.loc[df['Contract'] == 'Month-to-month', 'ContractStartDate'] = pd.Timestamp.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - pd.DateOffset(days=1)

print(df)


# In[25]:


# Extract a specific portion of a date column using dt accessor
df['ContractStartYear'] = df['ContractStartDate'].dt.year
# Extract a specific portion of a date column using strftime
df['ContractStartMonth'] = df['ContractStartDate'].apply(lambda x: x.strftime('%B'))


# In[28]:


#Pivot a dataframe using pivot_table
ddf = pd.read_csv('telecom_customer_churn.csv')
pivot = pd.pivot_table(ddf, values='TotalCharges', index='Contract', columns='PaymentMethod', aggfunc=np.sum)
pivot


# In[29]:


df = pd.read_csv('telecom_customer_churn.csv')
# Reshape a dataframe using melt
melted = pd.melt(df, id_vars=['customerID'], value_vars=['MonthlyCharges', 'TotalCharges'])
melted


# In[30]:


# Merge two dataframes based on common column names using join
df1 = pd.DataFrame({'customerID': [1, 2, 3], 'gender': ['Male', 'Female', 'Male']})
df2 = pd.DataFrame({'customerID': [2, 3, 4], 'age': [25, 30, 35]})
merged = df1.join(df2.set_index('customerID'), on='customerID')
merged


# In[31]:


# Merge two dataframes based on common column names using merge
df1 = pd.DataFrame({'customerID': [1, 2, 3], 'gender': ['Male', 'Female', 'Male']})
df2 = pd.DataFrame({'customerID': [2, 3, 4], 'age': [25, 30, 35]})
merged = pd.merge(df1, df2, on='customerID')
merged


# In[33]:


# Create a new column using a list comprehension
show =df['TenureMonthsList'] = [x*12 for x in df['tenure']]
show


# In[35]:


# Create a new column using np.select
conditions = [
    (df['MonthlyCharges'] > 100) & (df['tenure'] > 50),
    (df['MonthlyCharges'] > 50) & (df['tenure'] > 24),
    (df['MonthlyCharges'] < 50) & (df['tenure'] > 12)
]
values = ['High Value', 'Mid Value', 'Low Value']
display = df['CustomerValue'] = np.select(conditions, values)
display


# In[36]:


# Create a new column using cut
cut=df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=['1 year or less', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years'])
cut


# In[37]:


# Create a new column using qcut
qcut=df['MonthlyChargesGroup'] = pd.qcut(df['MonthlyCharges'], q=3, labels=['Low', 'Mid', 'High'])
qcut


# In[39]:


# Count the number of unique values in a column using nunique
print(df['PaymentMethod'].nunique())


# In[42]:


# Replace values in a column using replace
replace= df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
replace


# In[44]:


# Remove leading/trailing whitespace from values in a column using strip
d1= df['gender'] = df['gender'].str.strip()
d1


# In[45]:


# Capitalize the first letter of values in a column using str.title
d2=df['gender'] = df['gender'].str.title()
d2 


# In[46]:


# Remove special characters from values in a column using str.replace
d3=df['PaymentMethod'] = df['PaymentMethod'].str.replace('[^A-Za-z\s]+', '')
d3


# In[47]:


# Sort a dataframe using sort_values
sorted_df = df.sort_values(by=['MonthlyCharges', 'TotalCharges'], ascending=[False, True])
sorted_df 


# In[49]:


df = pd.read_csv('telecom_customer_churn.csv')
# Filter rows in a dataframe using boolean indexing
filtered = df[(df['gender'] == 'Male') & (df['tenure'] > 24)]
print(filtered)
# Drop rows from a dataframe using drop
df = df.drop([0, 1, 2])
print(df)
# Drop columns from a dataframe using drop
df = df.drop(['gender', 'SeniorCitizen'], axis=1)
print(df)
# Create a correlation matrix using corr
d4= corr_matrix = df.corr()
print(d4)


# In[50]:


# Create a heatmap of a correlation matrix using seaborn
import seaborn as sns
sns.heatmap(corr_matrix, annot=True)

# Create a scatterplot using matplotlib
import matplotlib.pyplot as plt
plt.scatter(df['MonthlyCharges'], df['TotalCharges'])
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.title('Scatterplot of Monthly Charges and Total Charges')


# In[51]:


# Create a histogram using matplotlib
plt.hist(df['tenure'], bins=10)
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.title('Histogram of Tenure')
plt.show()


# In[52]:


# Create a boxplot using matplotlib
plt.boxplot(df['MonthlyCharges'], vert=False)
plt.xlabel('Monthly Charges')
plt.title('Boxplot of Monthly Charges')
plt.show()


# In[53]:


# Create a pie chart using matplotlib
labels = ['Yes', 'No']
sizes = df['Churn'].value_counts()
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart of Customer Churn')
plt.show()


# In[55]:


# Create a line chart using matplotlib

x = df_sorted['Contract']
y = df_sorted['MonthlyCharges']
plt.plot(x, y)
plt.xlabel('Contract Start Date')
plt.ylabel('Monthly Charges')
plt.title('Line Chart of Monthly Charges by Contract Start Date')
plt.show()

