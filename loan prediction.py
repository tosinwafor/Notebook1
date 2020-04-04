
# coding: utf-8

# In[ ]:


#Pre Processing the Dataset
#Splitting the Input and the Target
#Standardizing the Dataset
#Splitting the Input and Target into Test and Train dataset
#Creating the Model
#Testing the Model


# In[613]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[614]:


loan = pd.read_csv('Loan_prediction.csv')


# In[615]:


loan


# In[616]:


loan.describe()


# In[617]:


loan.shape


# In[618]:


loan.columns


# In[619]:


loan.head()


# In[620]:


loan.tail()


# In[621]:


loan.info()


# In[622]:


loan.isnull()


# In[623]:


loan.isnull().sum()


# In[624]:


loan_missing = loan.isnull().sum()


# In[625]:


total_cells = np.product(loan.shape)
total_missing = loan_missing.sum()
(total_missing/total_cells) * 100


# In[626]:


loan.dropna(how='any').shape


# In[627]:


loan = pd.read_csv('Loan_prediction.csv')
loan = loan.drop('Loan_ID', axis = 1)


# In[628]:


loan.head()


# In[629]:


count = loan.Gender.value_counts()
count


# In[630]:


count.plot(kind='bar')
plt.title("Distribution of Gender, Male(1) and Female(0)")
plt.xlabel("Gender")
plt.ylabel("Count");


# In[631]:


count = loan.Loan_Status.value_counts()
count


# In[632]:


count.plot(kind='bar')
plt.title("Distribution of Loan Status, Yes(1) and No(0)")
plt.xlabel("Loan_Status")
plt.ylabel("Count");


# In[633]:


#our target class is Loan_Status


# In[634]:


count.plot(kind='bar')
plt.title("Distribution of Applicant's Income and Education")
plt.xlabel("ApplicantIncome")
plt.ylabel("Education");


# In[635]:


count.plot(kind='bar')
plt.title("Distribution of Loan_Status and Education")
plt.xlabel("Loan_Status")
plt.ylabel("Education");


# In[636]:


sns.countplot(x=loan['Education'],hue=loan['Loan_Status'])


# In[637]:


loan.drop('Loan_Status',  inplace= True, axis= 1)


# In[638]:


loan.head()


# In[639]:


loan.columns.values


# In[640]:


loan = loan.fillna(method = 'bfill', axis = 0, limit = 1) 


# In[641]:


loan


# In[642]:


loan.info()


# In[644]:


loan = pd.read_csv('Loan_prediction.csv')
sns.distplot(loan['ApplicantIncome'])


# In[645]:


value=(loan['ApplicantIncome']) & (loan['LoanAmount'])
loan['color']= np.where( value==True , "#9b59b6", "#3498db")

sns.regplot(data=loan, x="ApplicantIncome", y="LoanAmount", fit_reg=True, scatter_kws={'facecolors':loan['color']})


# In[646]:


sns.regplot(x=loan["ApplicantIncome"], y=loan["LoanAmount"], fit_reg=False, scatter_kws={'facecolors':loan['color']});


# In[647]:


loan['ApplicantIncome'].plot.hist(bins=20)


# In[648]:


loan.groupby(['Gender','Education'])['Loan_Status'].count()


# In[649]:


loan.groupby(['Gender','Education'])['Loan_Status'].count().plot(kind='bar')


# In[650]:


loan['Self_Employed'].value_counts(normalize=True)


# In[651]:


sns.countplot(loan['Self_Employed'],hue=loan['Loan_Status'])


# In[652]:


loan['Credit_History'].value_counts()


# In[653]:


loan.groupby(['Credit_History','Loan_Status'])['Loan_ID'].count()


# In[654]:


sns.countplot(loan['Credit_History'],hue=loan['Loan_Status'])


# In[655]:


loan['Property_Area'].value_counts().plot(kind='bar')


# In[656]:


sns.countplot(loan['Property_Area'],hue=loan['Loan_Status'])


# In[657]:


sns.distplot(loan['CoapplicantIncome'])


# In[658]:


loan = pd.read_csv('Loan_prediction.csv')
load_cols= [loan.select_dtypes(['Int64', 'Float64']).columns]


# In[659]:


loan.columns


# In[660]:


loan_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']


# In[661]:


plt.figure(figsize=(24, 18))
count = 1


# In[664]:


for cols in loan_cols:
    
    
    plt.subplot(3, 2, count)
    
    sns.boxplot(x='Loan_Status', y= cols, data= loan)
    
    count +=1


# In[668]:


for cols in loan_cols:
    plt.subplot(3, 2, count)
    
    sns.distplot(loan.loc[loan[cols].notna(), cols])
    
    count+=1
        
    

