# TITANIC SURVIVOR DATA : PREDICTION USING DIFFERENT ML MODELS (MAJOR PROJECT BY PARTH VAZE)


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```


```python
data=pd.read_excel('TITANIC.xlsx')
dxd=pd.read_excel('TITANIC.xlsx')
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>



# Variable Identification
## Dependent Variable 
Survived
## Independent Variable
1)Pclass
2)Sex
3)Age
4)Fare


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



## Data Types: 
### String:
1)Name

2)Sex

3)Ticket

4)Cabin

5)Embarked
### Numeric:
1)PassengerId

2)Survived

3)Age

4)SibSp

5)Parch

6)Fare

7)Pclass

## Step 2 : Univariate Analysis

### Analysis of independent and dependent variables taken one at a time : Countplots of each variable

#### A)Survival Data


```python
sns.countplot(x='Survived',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fefeb83588>




![png](output_11_1.png)


#### B)Sex data


```python
sns.countplot(x='Sex',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fefeb94248>




![png](output_13_1.png)


#### Embarked Port data


```python
sns.countplot(x='Embarked',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fefeb6e9c8>




![png](output_15_1.png)


#### Passenger Class data


```python
sns.countplot(x='Pclass',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fefeb6ce48>




![png](output_17_1.png)


#### Parch data


```python
sns.countplot(x='Parch',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff480fc8>




![png](output_19_1.png)


#### SibSp data


```python
sns.countplot(x='SibSp',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff4c4448>




![png](output_21_1.png)


#### Age data


```python
sns.boxplot(y='Age',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff54a348>




![png](output_23_1.png)



```python
sns.distplot(data['Age'],bins=5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff5c5b88>




![png](output_24_1.png)


# Step 3 : Bivariate Analysis
## Continous Data :
1)Age

2)Fare

## Categorical :
1)Sex

2)Pclass

3)Survived

4)Embarked


```python
sns.countplot(x='Sex',data=data,hue='Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff54a888>




![png](output_26_1.png)



```python
counts=data.groupby(['Sex','Survived'],axis=0)
counts.size()
```




    Sex     Survived
    female  0            81
            1           233
    male    0           468
            1           109
    dtype: int64



Females had a higher survival rate as compared to males


```python
sns.countplot(x='Pclass',data=data,hue='Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff6a8f88>




![png](output_29_1.png)



```python
counts=data.groupby(['Pclass','Survived'],axis=0)
counts.size()
```




    Pclass  Survived
    1       0            80
            1           136
    2       0            97
            1            87
    3       0           372
            1           119
    dtype: int64



The first class passengers had the highest survival rate , followed by the 2nd class passengers and then by the 3rd class passengers


```python
sns.countplot(x='Embarked',data=data,hue='Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff72c888>




![png](output_32_1.png)



```python
counts=data.groupby(['Embarked','Survived'],axis=0)
counts.size()
```




    Embarked  Survived
    C         0            75
              1            93
    Q         0            47
              1            30
    S         0           427
              1           217
    dtype: int64



The people who embarked from port S had the worst survival rate while the people embarking from port C had the highest survival rate


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    

# Step 4 :Missing value treatment


```python
data['Age']
```




    0      22.0
    1      38.0
    2      26.0
    3      35.0
    4      35.0
           ... 
    886    27.0
    887    19.0
    888     NaN
    889    26.0
    890    32.0
    Name: Age, Length: 891, dtype: float64




```python
sns.distplot(data['Age'],bins=5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff8213c8>




![png](output_38_1.png)



```python
data['Age'].describe()
```




    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: Age, dtype: float64




```python
data['Age'].isnull().sum()
```




    177



The age data is skewed as seen in the histogram and also the type of skewness is left skewed as median>mean

Null values of age are 177 and so these cannot be deleted as they are far greater than 5 percent of our data

To treat these missing values we will have to fill them with the median value of the age data as the graph is skewed


```python
data['Age']=data['Age'].fillna(value=data['Age'].median())
```


```python
data['Age']
```




    0      22.0
    1      38.0
    2      26.0
    3      35.0
    4      35.0
           ... 
    886    27.0
    887    19.0
    888    28.0
    889    26.0
    890    32.0
    Name: Age, Length: 891, dtype: float64




```python
data['Age'].isnull().sum()
```




    0



We have now filled the missing values in the age data with median for the data


```python
data['Age'].describe()
```




    count    891.000000
    mean      29.361582
    std       13.019697
    min        0.420000
    25%       22.000000
    50%       28.000000
    75%       35.000000
    max       80.000000
    Name: Age, dtype: float64




```python
data['Age']
```




    0      22.0
    1      38.0
    2      26.0
    3      35.0
    4      35.0
           ... 
    886    27.0
    887    19.0
    888    28.0
    889    26.0
    890    32.0
    Name: Age, Length: 891, dtype: float64




```python
sns.boxplot(y='Age',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff9eb808>




![png](output_48_1.png)


The age data has outliers which might cause bias in our data.

Therefore we should cap the outlier data to the 95th percentile at the top and 5th percentile at the bottom


```python
data['Age']
```




    0      22.0
    1      38.0
    2      26.0
    3      35.0
    4      35.0
           ... 
    886    27.0
    887    19.0
    888    28.0
    889    26.0
    890    32.0
    Name: Age, Length: 891, dtype: float64



I am identifying the outliers by using the Interquartile range

Any value, which is beyond the range of -1.5x IQR to 1.5 x IQR where IQR = Q3-Q1 : Treated as outlier


```python
IQR = data['Age'].quantile(0.75)-data['Age'].quantile(0.25)
print(IQR)
```

    13.0
    


```python
ul = data['Age'].quantile(0.75) + 1.5*IQR
ll = data['Age'].quantile(0.25) - 1.5*IQR
print("Upper cap",ul)
print("Lower cap",ll)
```

    Upper cap 54.5
    Lower cap 2.5
    

For treating the outlier values we will now cap it to 95th percentile for age and 5th percentile for the fare


```python
y1=data['Age']
y1
```




    0      22.0
    1      38.0
    2      26.0
    3      35.0
    4      35.0
           ... 
    886    27.0
    887    19.0
    888    28.0
    889    26.0
    890    32.0
    Name: Age, Length: 891, dtype: float64




```python
for x in range(891):
    if y1[x]>=ul:
        y1[x]=y1.quantile(.95)
    elif y1[x]<=ll:
        y1[x]=y1.quantile(.05)
```

    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    


```python
y1.describe()
```




    count    891.000000
    mean      29.109989
    std       11.826281
    min        3.000000
    25%       22.000000
    50%       28.000000
    75%       35.000000
    max       54.000000
    Name: Age, dtype: float64




```python
data['Age']=y1
```


```python
data['Age']
```




    0      22.0
    1      38.0
    2      26.0
    3      35.0
    4      35.0
           ... 
    886    27.0
    887    19.0
    888    28.0
    889    26.0
    890    32.0
    Name: Age, Length: 891, dtype: float64




```python
sns.boxplot(y='Age',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff85a408>




![png](output_60_1.png)


### Data of age after capping looks like follows in the boxplot


```python

sns.boxplot(y='Age',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feff9fed08>




![png](output_62_1.png)


The age data has been treated and now contains no outliers

# Step 6 : Feature Engineering
## Variable and Dummy Variable creation
To simplify our data analysis we will now create the dummy variables which will allow us to get the required information easily and by using lesser variables

We will also delete the unnecessary variables from our dataset

In this data , name ,ticket,passenger id, and cabin are not useful to us and so will be deleted.

We will also add 5 new columns ("C Or A" , "W/C" , "CL/S" , "CL/AG" and "CL/AG/SE") to our data to do our analysis on a deeper level


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>28.0</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>




```python
obj=data.dtypes==np.object
obj
```




    PassengerId    False
    Survived       False
    Pclass         False
    Name            True
    Sex             True
    Age            False
    SibSp          False
    Parch          False
    Ticket          True
    Fare           False
    Cabin           True
    Embarked        True
    dtype: bool




```python
data.columns[obj]
```




    Index(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')




```python

del data['Name']
del data['Ticket']
del data['Cabin']
```


```python
del data['PassengerId']
```


```python
obj=data.dtypes==np.object
```


```python
obj
```




    Survived    False
    Pclass      False
    Sex          True
    Age         False
    SibSp       False
    Parch       False
    Fare        False
    Embarked     True
    dtype: bool




```python
data.columns[obj]
```




    Index(['Sex', 'Embarked'], dtype='object')




```python
dummydf = pd.DataFrame()
for i in data.columns[obj]:
    dummy=pd.get_dummies(data[i], drop_first=True)
    dummydf=pd.concat([dummydf, dummy], axis=1)
print(dummydf)
```

         male  Q  S
    0       1  0  1
    1       0  0  0
    2       0  0  1
    3       0  0  1
    4       1  0  1
    ..    ... .. ..
    886     1  0  1
    887     0  0  1
    888     0  0  1
    889     1  0  0
    890     1  1  0
    
    [891 rows x 3 columns]
    


```python
d1=pd.concat([data,dummydf], axis=1)
```


```python
d1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>male</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>28.0</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 11 columns</p>
</div>




```python
d1.drop(['Sex','Embarked'], axis=1, inplace=True)
```


```python
d2 = pd.get_dummies(data, drop_first=True)
```


```python
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Sex_male</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 9 columns</p>
</div>



Now fare data is representative of the class of the passengers and therefore is reduntant. Thus we can remove it from our analysis


```python
del d2['Fare']
```


```python
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Sex_male</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 8 columns</p>
</div>



Besides the fare data, we can see that the SibSp and Parch variables add no considerable value to our data

Therefore we will now proceed to remove them


```python
del d2['SibSp']
del d2['Parch']
```


```python
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Sex_male</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 6 columns</p>
</div>



## Reasoning :

I will now add the first column "C Or A" to the d2 data frame 

This column is created to differentiate between children and adults using the age parameter 

I have considered passenger with age less than 18 as children while greater than or equal to 18 as adults

In this column --> A: Adult and C:Child



```python
a=d2['Age']
```


```python
d2.insert(4,"C Or A",0)

for i in range(891):
    if(a[i]>0 and a[i]<18):
        d2['C Or A'][i]='C' 
    elif(a[i]>=18):
        d2['C Or A'][i]='A'
```

    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys
    C:\Users\ssvaz\anaconda3\lib\site-packages\pandas\core\indexing.py:670: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_with_indexer(indexer, value)
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    

## Reasoning:

Next , I am adding the column "W/C" to the data , which describes whether the passenger is an adult or child as well as whether the passenger is male or female

In this column --> WC: Female(Women) Child , MA : Adult Male , WA : Female(Women) Adult and MC : Male Child


```python
g=d2['Sex_male']
a1=d2['C Or A']
```


```python
d2.insert(5,'W/C','')
for i in range(891):
    if(g[i]==0 and a1[i]=='C'):
        d2['W/C'][i]='WC'
    elif(g[i]==1 and a1[i]=='A'):
        d2['W/C'][i]='MA'
    elif(g[i]==0 and a1[i]=='A'):
        d2['W/C'][i]='WA'
    elif (g[i]==1 and a1[i]=='C'):
        d2['W/C'][i]='MC'        
```

    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    

## Reasoning:

Now I will be creating a column to categorize the passengers by their passenger class as well as their sex

This column will allow me to do bivariate analysis of the 2 variables with respect to the survival data


```python
s=d2['Sex_male']
c=d2['Pclass']
```


```python
d2.insert(6,'CL/S','')
for i in range(891):
    if(s[i]==0 and c[i]==3):
        d2['CL/S'][i]='3F'
    elif(s[i]==1 and c[i]==3):
        d2['CL/S'][i]='3M'
    elif(s[i]==0 and c[i]==2):
        d2['CL/S'][i]='2F'
    elif (s[i]==1 and c[i]==2):
        d2['CL/S'][i]='2M'
    elif (s[i]==0 and c[i]==1):
        d2['CL/S'][i]='1F'
    elif (s[i]==1 and c[i]==1):   
        d2['CL/S'][i]='1M'
```

    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    

## Reasoning:

Now I will be creating a column to categorize the passengers by their passenger class as well as their age class

This column will allow me to do bivariate analysis of the 2 variables with respect to the survival data of the respective passengers


```python
ag=d2['C Or A']
c=d2['Pclass']

```


```python
d2.insert(7,'CL/AG','')
for i in range(891):
    if(ag[i]=='A' and c[i]==3):
        d2['CL/AG'][i]='3Ad'
    elif(ag[i]=='C' and c[i]==3):
        d2['CL/AG'][i]='3Ch'
    elif(ag[i]=='A' and c[i]==2):
        d2['CL/AG'][i]='2Ad'
    elif (ag[i]=='C' and c[i]==2):
        d2['CL/AG'][i]='2Ch'
    elif (ag[i]=='A' and c[i]==1):
        d2['CL/AG'][i]='1Ad'
    elif (ag[i]=='C' and c[i]==1):   
        d2['CL/AG'][i]='1Ch'
```

    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    

## Reasoning:

Now I will be creating a column to categorize the passengers by their passenger class , their sex and their age class

This column will allow me to do trivariate analysis of the 3 variables with respect to the survival data of the respective passengers


```python

ag=d2['C Or A']
c=d2['Pclass']
s1=d2['Sex_male']
```


```python

d2.insert(8,'CL/AG/GE','')
for i in range(891):
    if(ag[i]=='A' and c[i]==3 and s1[i]==0):
        d2['CL/AG/GE'][i]='3AdF'
    elif(ag[i]=='A' and c[i]==3 and s1[i]==1):
        d2['CL/AG/GE'][i]='3AdM'
    elif(ag[i]=='C' and c[i]==3 and s1[i]==1):
        d2['CL/AG/GE'][i]='3ChM'
    elif(ag[i]=='C' and c[i]==3 and s1[i]==0):
         d2['CL/AG/GE'][i]='3ChF'       
    elif(ag[i]=='A' and c[i]==2 and s1[i]==0):
        d2['CL/AG/GE'][i]='2AdF'
    elif(ag[i]=='A' and c[i]==2 and s1[i]==1):
        d2['CL/AG/GE'][i]='2AdM'
    elif (ag[i]=='C' and c[i]==2 and s1[i]==1):
        d2['CL/AG/GE'][i]='2ChM'
    elif (ag[i]=='C' and c[i]==2 and s1[i]==0):
        d2['CL/AG/GE'][i]='2ChF'
    elif(ag[i]=='A' and c[i]==1 and s1[i]==0):
        d2['CL/AG/GE'][i]='1AdF'
    elif(ag[i]=='A' and c[i]==1 and s1[i]==1):
        d2['CL/AG/GE'][i]='1AdM'
    elif (ag[i]=='C' and c[i]==1 and s1[i]==1):
        d2['CL/AG/GE'][i]='1ChM'
    elif (ag[i]=='C' and c[i]==1 and s1[i]==0):
        d2['CL/AG/GE'][i]='1ChF'
    
```

    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:20: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:22: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      app.launch_new_instance()
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:26: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    


# Step 7 : Summary of Analysis and Treatment of the data

## Original Data : TITANIC


```python
dxd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>



The modified data used for the analysis


```python
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Sex_male</th>
      <th>C Or A</th>
      <th>W/C</th>
      <th>CL/S</th>
      <th>CL/AG</th>
      <th>CL/AG/GE</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>2M</td>
      <td>2Ad</td>
      <td>2AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>1M</td>
      <td>1Ad</td>
      <td>1AdM</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 11 columns</p>
</div>



### Step 1 : Treatment of missing values in Age column


```python
dxd['Age'].isnull().sum()
```




    177




```python
d2['Age'].isnull().sum()
```




    0




```python
sns.distplot(data['Age'],bins=5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feffb120c8>




![png](output_108_1.png)


dxd is the original data with 177 null values for the age column

d2 is the data frame which has the null values for the age column filled with the central values 

In this case the central value used is median as the age data is right skewed and so using mean would have reduced the accuracy of our analysis

## Step 2 : Treatment of outliers in the age data


```python
sns.boxplot(y='Age',data=dxd)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feffc0eec8>




![png](output_111_1.png)


The original data(dxd) for age had outliers which had to be capped using the 95th and 5th quantile for the upper and lower outliers respectively

The data after capping(d2) looks as follows


```python
sns.boxplot(y='Age',data=d2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feffc6b5c8>




![png](output_113_1.png)


The capped data has no outliers and can be used for further data analysis

## Step 3 : Gauging the effect of the various data points with respect to the survival rate

### A)Correlation of survival rate with the passenger class


```python
sns.countplot(x='Pclass',data=d2,hue='Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feffcbedc8>




![png](output_116_1.png)


The above graph clearly indicates that the survival was the highest for the 1st class passengers and lowest for the 3rd class passengers

This is in line with the fact that the 1st class passengers(Rich passengers) were given preference over the 2nd and 3rd class passengers during the boarding of the lifeboats.


```python
counts3 = d2.groupby(['Pclass', 'Survived'], axis= 0)
counts3.size()
```




    Pclass  Survived
    1       0            80
            1           136
    2       0            97
            1            87
    3       0           372
            1           119
    dtype: int64



 The above numbers also support the aforementioned analysis

### B)Determining relation between the port of embarkment and the survival rate (if any)


```python
sns.countplot(x='Embarked',data=data,hue='Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feffd2ee88>




![png](output_121_1.png)


The above graph leads us to infer that the rate of survival for those embarked from port C was highest of all the 3 ports

This inference might be just a coincidence or plain luck but still I have included it in this analysis as an extra observation


```python
counts4=data.groupby(['Embarked','Survived'],axis=0)
counts4.size()
```




    Embarked  Survived
    C         0            75
              1            93
    Q         0            47
              1            30
    S         0           427
              1           217
    dtype: int64



The above data supports the aforementioned observation

### C)Correlating age and survival rate

To find out this correlation we first need to set the criteria for the age limit for children

For this data , I have assumed that if the age of the passenger is less than 18 , then the passenger is a child.

If his or her age is equal to or above 18 , then I have considered him an adult.

I have then created a column in the data frame to store if the passenger is an adult or a child


```python
d2['C Or A'].describe()
```




    count     891
    unique      2
    top         A
    freq      778
    Name: C Or A, dtype: object




```python
sns.countplot(x='C Or A',data=d2,hue='Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feffdaf188>




![png](output_128_1.png)


# Legend :
A-Adult C-Child

This graph shows us that the survival rate for children was far higher than that for the adults

## D)Correlation between the survival rate and the passenger class and age class taken together


```python
sns.countplot(x='CL/AG',data=d2,hue='Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feffde7208>




![png](output_131_1.png)


### Legend :

Ad-Adult Ch-Child

1,2 and 3 are the passenger classes

The graph clearly shows that the worst survival rate was for the 3rd class adult passengers and the better survival rate was for the 1st class children passengers

This obvservation is again in line with preferential treatment given to the 1st class passengers along with that given to children


```python
counts5=d2.groupby(['CL/AG','Survived'],axis=0)
counts5.size()
```




    CL/AG  Survived
    1Ad    0            79
           1           125
    1Ch    0             1
           1            11
    2Ad    0            95
           1            66
    2Ch    0             2
           1            21
    3Ad    0           323
           1            90
    3Ch    0            49
           1            29
    dtype: int64



The above data supports the aforementioned problem

## E)Correlation between the survival rate and passenger class and the gender taken together


```python
sns.countplot(x='CL/S',data=d2,hue='Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1feffe98e08>




![png](output_137_1.png)


## Legend:

M - Male F - Female

1,2 and 3 are the passenger classes

This graph clearly shows that 1st class female passengers had best survival rate while survival rate was the worst for 3rd class male passengers 

This again points to the fact that 1st class and female passengers were the first to be boarded onto the lifeboats which inturn greatky increased their survival chances

## F)Correlation between the survival rate and passenger class,the gender and the age taken  together


```python
sns.set(rc={'figure.figsize':(11,7)})
sns.countplot(x='CL/AG/GE',data=d2,hue='Survived',dodge=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fefff08d08>




![png](output_140_1.png)


## Legend:

AdF-Adult Female

AdM-Adult Male

ChM-Male Child

ChF-Female Child

1,2 and 3 are the passenger classes

This graph compares the survival rate across the passenger class, gender and their age.
From this we can clearly see that 3rd class adult male passenegrs were the ones with worst survival rate while most of the 1st class child passengers of both genders survived the sinking.

The below data gives the numerical proof for this graph


```python
counts6=d2.groupby(['CL/AG/GE','Survived'],axis=0)
print(counts6.size())
```

    CL/AG/GE  Survived
    1AdF      0             2
              1            84
    1AdM      0            77
              1            41
    1ChF      0             1
              1             7
    1ChM      1             4
    2AdF      0             6
              1            58
    2AdM      0            89
              1             8
    2ChF      1            12
    2ChM      0             2
              1             9
    3AdF      0            56
              1            53
    3AdM      0           267
              1            37
    3ChF      0            16
              1            19
    3ChM      0            33
              1            10
    dtype: int64
    

## Finally we will have following cleaned and complete data set :


```python
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Sex_male</th>
      <th>C Or A</th>
      <th>W/C</th>
      <th>CL/S</th>
      <th>CL/AG</th>
      <th>CL/AG/GE</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>2M</td>
      <td>2Ad</td>
      <td>2AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>1M</td>
      <td>1Ad</td>
      <td>1AdM</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 11 columns</p>
</div>



### The legend for the columns is as follows:

1)Survived : 0 for not survived and 1 for survived
Ad-Adult Ch-Child

1,2 and 3 are the passenger classes
2)Pclass : Passenger class (1,2 or 3)

3)Age : Passenger age

4)Sex_male(Dummy Variable) : Describes whether passeneger is male or not (1:male 0:Female)

5)C Or A : Points out whether passenger is an adult or a child (A:Adult C:Child)

6)W/C : W:Female(woman) M:Male(man)

7)CL/S : F:Female(woman) M:Male(man) and the classes are 1,2 and 3

8)CL/AG : Ad-Adult Ch-Child ; 1,2 and 3 are the passenger classes

9)Embarked_Q , Embarked_S are the dummy variables which tell us the port of Embarkment(1:True,0:False)

# Model Fitting and Evaluation

## As the data is labelled data , I have used supervised learning algorithms for the model fitting :

**1)Logistic Regression**

**2)Decision Tree**

## 1)Logistic Regression

### Displaying the data


```python
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Sex_male</th>
      <th>C Or A</th>
      <th>W/C</th>
      <th>CL/S</th>
      <th>CL/AG</th>
      <th>CL/AG/GE</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>2M</td>
      <td>2Ad</td>
      <td>2AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>0</td>
      <td>A</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>1M</td>
      <td>1Ad</td>
      <td>1AdM</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>1</td>
      <td>A</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 11 columns</p>
</div>



### A) Using the Sex_male column data as the independent X data


```python
x=d2.iloc[:,3:4].values
x.shape

```




    (891, 1)




```python
y=d2.iloc[:,0].values
y.shape
```




    (891,)




```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=5)
```


```python
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
y_p=lr.predict(x_test)
y_p
```




    array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
           1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
           0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,
           0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
           0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 1], dtype=int64)




```python
y_test
```




    array([0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1,
           1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1,
           0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
           0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
           0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
           0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
           0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
           1, 1, 1], dtype=int64)



**Confusion Matrix**


```python
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,y_p)
c
```




    array([[125,  15],
           [ 27,  56]], dtype=int64)



**Classification Report**


```python
from sklearn.metrics import classification_report as cr
print(cr(y_test,y_p))
```

                  precision    recall  f1-score   support
    
               0       0.82      0.89      0.86       140
               1       0.79      0.67      0.73        83
    
        accuracy                           0.81       223
       macro avg       0.81      0.78      0.79       223
    weighted avg       0.81      0.81      0.81       223
    
    

## B)Using the age data as independent X data

**I am converting the C Or A column of the data frame to numerical value to fit it into the logistic regression model**


```python
d2.rename(columns={'C Or A':'Age_Adult'},inplace=True)
c1=d2['Age_Adult']
for i in range(891):
    if(c1[i]=='A'):
        c1[i]=1
    elif(c1[i]=='C'):
        c1[i]=0

d2
```

    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    C:\Users\ssvaz\anaconda3\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Sex_male</th>
      <th>Age_Adult</th>
      <th>W/C</th>
      <th>CL/S</th>
      <th>CL/AG</th>
      <th>CL/AG/GE</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>2M</td>
      <td>2Ad</td>
      <td>2AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>1M</td>
      <td>1Ad</td>
      <td>1AdM</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 11 columns</p>
</div>




```python
d2['Age_Adult']=d2['Age_Adult'].astype(int)

```

## Displaying the data


```python
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Sex_male</th>
      <th>Age_Adult</th>
      <th>W/C</th>
      <th>CL/S</th>
      <th>CL/AG</th>
      <th>CL/AG/GE</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>2M</td>
      <td>2Ad</td>
      <td>2AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>1M</td>
      <td>1Ad</td>
      <td>1AdM</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 11 columns</p>
</div>




```python
x1=d2.iloc[:,4:5].values
x1.shape
```




    (891, 1)




```python
y1=d2.iloc[:,0].values
y1.shape
```




    (891,)




```python
from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.25,random_state=5)
```


```python
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
y1_p=lr.predict(x1_test)
y1_p
```




    array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
           0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0], dtype=int64)



y1_test


```python
from sklearn.metrics import confusion_matrix
c1=confusion_matrix(y1_test,y1_p)
c1
```




    array([[125,  15],
           [ 69,  14]], dtype=int64)



**Classification Report**


```python
from sklearn.metrics import classification_report as cr
print(cr(y1_test,y1_p))
```

                  precision    recall  f1-score   support
    
               0       0.64      0.89      0.75       140
               1       0.48      0.17      0.25        83
    
        accuracy                           0.62       223
       macro avg       0.56      0.53      0.50       223
    weighted avg       0.58      0.62      0.56       223
    
    

## As is clear from the classification reports below , the best accuracy is obtained for the sex data for the logistic regression model

###  Classification report for the logistic regression model where X=sex_male column data


```python
from sklearn.metrics import classification_report as cr
print(cr(y_test,y_p))
```

                  precision    recall  f1-score   support
    
               0       0.82      0.89      0.86       140
               1       0.79      0.67      0.73        83
    
        accuracy                           0.81       223
       macro avg       0.81      0.78      0.79       223
    weighted avg       0.81      0.81      0.81       223
    
    

###  Classification report for the logistic regression model where X=Age_Adult column data


```python
from sklearn.metrics import classification_report as cr
print(cr(y1_test,y1_p))
```

                  precision    recall  f1-score   support
    
               0       0.64      0.89      0.75       140
               1       0.48      0.17      0.25        83
    
        accuracy                           0.62       223
       macro avg       0.56      0.53      0.50       223
    weighted avg       0.58      0.62      0.56       223
    
    

## Decision Tree

### Displaying the data


```python
d2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Sex_male</th>
      <th>Age_Adult</th>
      <th>W/C</th>
      <th>CL/S</th>
      <th>CL/AG</th>
      <th>CL/AG/GE</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>2M</td>
      <td>2Ad</td>
      <td>2AdM</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>1F</td>
      <td>1Ad</td>
      <td>1AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>28.0</td>
      <td>0</td>
      <td>1</td>
      <td>WA</td>
      <td>3F</td>
      <td>3Ad</td>
      <td>3AdF</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>1M</td>
      <td>1Ad</td>
      <td>1AdM</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32.0</td>
      <td>1</td>
      <td>1</td>
      <td>MA</td>
      <td>3M</td>
      <td>3Ad</td>
      <td>3AdM</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 11 columns</p>
</div>



### A)Using the Age_Adult data as the independent X data


```python
x2=d2.iloc[:,4:5].values
x2.shape
```




    (891, 1)




```python
y2=d2.iloc[:,0].values
y2.shape
```




    (891,)




```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, random_state=1)

```


```python
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(x2_train,y2_train)
#Predict the response for test dataset
y2_pred = clf.predict(x2_test)
y2_pred
```




    array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           0, 0, 0, 0], dtype=int64)




```python
y2_test
```




    array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
           1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0,
           1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
           1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
           0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
           1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
           0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1,
           0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0,
           1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1,
           0, 0, 1, 0], dtype=int64)




```python
from sklearn import metrics
```


```python
print("Accuracy:",round(metrics.accuracy_score(y2_test, y2_pred)*100,2),"%")
```

    Accuracy: 57.84 %
    


```python
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y2_test,y2_pred)
```


```python
cm
```




    array([[136,  17],
           [ 96,  19]], dtype=int64)




```python
from sklearn.metrics import classification_report as cr
print(cr(y2_test,y2_pred))
```

                  precision    recall  f1-score   support
    
               0       0.59      0.89      0.71       153
               1       0.53      0.17      0.25       115
    
        accuracy                           0.58       268
       macro avg       0.56      0.53      0.48       268
    weighted avg       0.56      0.58      0.51       268
    
    

Accuracy of this model for the age data is low and so now let us try the model with the sex data

### B)Using the Sex_male data as the independent X data


```python
x3=d2.iloc[:,3:4].values
x3.shape
```




    (891, 1)




```python
y3=d2.iloc[:,0].values
y3.shape
```




    (891,)




```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.3, random_state=1)
```


```python
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(x3_train,y3_train)
#Predict the response for test dataset
y3_pred = clf.predict(x3_test)
y3_pred
```




    array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
           1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
           1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
           0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0,
           0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1,
           0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
           0, 0, 0, 0], dtype=int64)




```python
y3_test
```




    array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
           1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0,
           1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
           1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
           0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
           1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
           0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1,
           0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0,
           1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1,
           0, 0, 1, 0], dtype=int64)




```python
print("Accuracy:",round(metrics.accuracy_score(y3_test, y3_pred)*100,2),"%")
```

    Accuracy: 75.37 %
    


```python
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y3_test,y3_pred)
```


```python
cm
```




    array([[129,  24],
           [ 42,  73]], dtype=int64)




```python
from sklearn.metrics import classification_report as cr
print(cr(y3_test,y3_pred))
```

                  precision    recall  f1-score   support
    
               0       0.75      0.84      0.80       153
               1       0.75      0.63      0.69       115
    
        accuracy                           0.75       268
       macro avg       0.75      0.74      0.74       268
    weighted avg       0.75      0.75      0.75       268
    
    

## As is evident from the classification reports given below, the accuracy of the Decision tree model is more for the sex_male data


```python
from sklearn.metrics import classification_report as cr
print(cr(y2_test,y2_pred))
```

                  precision    recall  f1-score   support
    
               0       0.59      0.89      0.71       153
               1       0.53      0.17      0.25       115
    
        accuracy                           0.58       268
       macro avg       0.56      0.53      0.48       268
    weighted avg       0.56      0.58      0.51       268
    
    


```python
from sklearn.metrics import classification_report as cr
print(cr(y3_test,y3_pred))
```

                  precision    recall  f1-score   support
    
               0       0.75      0.84      0.80       153
               1       0.75      0.63      0.69       115
    
        accuracy                           0.75       268
       macro avg       0.75      0.74      0.74       268
    weighted avg       0.75      0.75      0.75       268
    
    


```python
print("The best accuracy is obtained using the sex_male data in the logistic regression model")
print("Accuracy:",round(metrics.accuracy_score(y_test, y_p)*100,2),"%")
```

    The best accuracy is obtained using the sex_male data in the logistic regression model
    Accuracy: 81.17 %
    

## Conclusion:
**The best model for this problem is the logistic regression model and the data used should be the sex_male column data**
