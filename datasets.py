
"""
Includes classes for every dataset that load and preprocess them.
"""

import pandas as pd
import os, datetime, re
import utils

# Path to the folder containing the dataset files
DATA_PATH="datasets/"

MIN_SAMPLES_PER_ISLAND=500
RANDOM_STATE=1

class Dataset:

   @property
   def preprocessed(self):
      return self.preprocess(self.load_raw())
   
   @property
   def natural_partition(self):
      return self.get_natural_partition(
         df=self.preprocessed,
         partition_on_column=self.partition_on_column,
         one_hot_encoded=False,
         drop_after=True
      )
   
   @property
   def X_y(self):
      # X and y should be preprocessed globally
      df=self.preprocessed

      # If we haven't removed the column we partition on, do so.
      if hasattr(self,'partition_on_column') and self.partition_on_column in df.columns:
         df=df.drop(columns=[self.partition_on_column])

      # Divide them and return
      return self.divide_X_y(df)
   
   def get_natural_partition(self,df,partition_on_column,one_hot_encoded,drop_after=True):
      # We assume it already has been preprocessed

      if not one_hot_encoded:
         # Make sure every island has at leas [] examples. Else drop them.
         t=df[partition_on_column].value_counts()[(df[partition_on_column].value_counts()>MIN_SAMPLES_PER_ISLAND)].index.to_list()
         df=df.loc[df[partition_on_column].isin(t)]

         # Unique values to partition on
         u_values=df[partition_on_column].unique()

         # Save partition indices
         partition_indices={g:df[df[partition_on_column]==g].index for g in u_values}


         # Drop column to partition on
         df=df.drop(columns=[partition_on_column])

      else:
         # Column names for one-hot-code encoded column
         cols=[col for col in df.columns if partition_on_column in col]

         # Check only that column names only have 1 underscore as we will split on it
         for col in cols:
            assert len(col.split('_'))==2,f'Column names cannot have underscores: {col}'

         # Obtain the partition indices
         partition_indices={col.split('_')[1]:df[df[col==1]].index for col in cols}
         
         df=df.drop(columns=[cols])         

      # And construct the partition. group -> arr of corr. indices (X,y)
      partition={}
      for group,ixs in partition_indices.items():
         partition_df=df.loc[ixs]
         X,y=self.divide_X_y(partition_df)
         partition[group]=(X,y)

      # Make sure partition does not alter index
      check_partition(partition,df)

      return partition
   
   def load_raw(self):
      return pd.read_csv(self.fpath)
   
   def divide_X_y(self,df):
      y=df[self.y_column]
      X=df.drop(self.y_column,axis=1)
      return X,y
   
   def __repr__(self):
      name=str(self.__class__.__name__).split("Dataset")[0]
      return ' '.join(re.findall('[A-Z][^A-Z]*',name))


class CensusIncome(Dataset): # TODO add 'Dataset'
   """
   Source: https://archive.ics.uci.edu/ml/datasets/census+income
   """

   def __init__(self,fname=f'{DATA_PATH}/adult.data'):
      # Path to data
      self.fpath=fname

      # Column to predict
      self.y_column='income'

      # Column to partition on
      self.partition_on_column='native-country'

      # Whether the task is 'regression' or 'classification'
      self.task='classification'

   def load_raw(self):
      # Column names to load
      names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship',
         'race','sex','capital-gains','capital-loss','hours-per-week','native-country','income']

      # Read from csv without index column, encoding=latin1 and na_values='?'
      return pd.read_csv(self.fpath,names=names,index_col=False,encoding='latin1',na_values='?',skipinitialspace=True)

   def preprocess(self,df):
      """
      Preprocessing leaves self.partition_on_column intact.
      """
      # Bin age into 5 equally sized buckets 
      ageBins=pd.cut(df['age'],bins=5)
      df=df.drop('age',axis=1)
      df['age']=ageBins

      # One-hot encodings
      one_hot_cols=['age','education','occupation','sex',
      'race','relationship','marital-status','workclass']
      # Check if they're in df
      one_hot_cols=[i for i in one_hot_cols if i in df.columns]
      df=pd.get_dummies(df,columns=one_hot_cols)

      # Drop irrelevant columns
      df=df.drop(['fnlwgt'],axis=1)

      # Replace y_column for categorical codes
      df['income']=pd.Categorical(df['income']).codes

      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True) 

      return df

   def get_natural_partition(self,df,partition_on_column,one_hot_encoded,drop_after):
      # Group countries into larger groups
      replacements={
         'latin-america':['Mexico','Puerto-Rico','El-Salvador','Cuba','Jamaica','Dominican-Republic','Guatemala','Columbia','Haiti','Nicaragua','Peru','Ecuador','Honduras'],
         'asia':['Philippines','India','China','Vietnam','Japan','Taiwan','Hong','Thailand'],
         'uk':['England','Scotland','Ireland'],
         'europe':['Germany','Italy','Poland','Portugal','Greece','France','Hungrary'],
         'us':['United-States']
      }
      for value,to_replace in replacements.items():
         df['native-country']=df['native-country'].replace(to_replace,value)

      # Include only those groups
      df=df[df['native-country'].isin(replacements.keys())]
      df=df.reset_index(drop=True) # Reset index before partitioning

      return super().get_natural_partition(df,partition_on_column,one_hot_encoded,drop_after)
   

class VehicleDataset(Dataset):
   """
   Source: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
   """

   def __init__(self,fname=f'{DATA_PATH}/vehicles.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='price'

      # Column to partition on
      self.partition_on_column='state'
      
      # Whether the task is 'regression' or 'classification'
      self.task='regression'
  
   def preprocess(self,df):
      # Distracting columns - exclude 'state' as we will partition on it
      cols_to_drop=['id', 'model','url', 'region', 'region_url', 'VIN', 'image_url', 'description',
                    'county', 'lat', 'long', 'posting_date']
      
      # Columns to one-hot-code encode
      one_hot_cols=['manufacturer','condition','fuel','title_status','transmission','drive','size',
                    'type','paint_color']
      
      # Drop distracting columns
      df=df.drop(columns=cols_to_drop)

      # Drop rows with NaN values
      df=df.dropna()

      # Cylinders cleanup
      df=df.drop(df[df['cylinders']=='other'].index)
      df['cylinders']=df['cylinders'].str.split(expand=True)[0].astype('int64')

      # Make year relative
      df['year']=datetime.datetime.now().year-df['year']

      # Delete price and odometer column outliers
      outlier_cols=['price','odometer']
      for col in outlier_cols:
         upper,lower = df[col].quantile(0.99), df[col].quantile(0.1)
         df = df[(df[col] < upper) & (df[col] > lower)]

      # TEMP
      t=df['state'].value_counts()[(df['state'].value_counts()>=2000)].index.to_list()
      df=df.loc[df['state'].isin(t)]

      # One-hot-code encodings
      df=pd.get_dummies(df,columns=one_hot_cols)

      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True) 

      return df
   
class BlackFridayDataset(Dataset):
   """
   Source: https://www.kaggle.com/datasets/sdolezel/black-friday
   """

   def __init__(self,fname=f'{DATA_PATH}/black_friday.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='Purchase'

      # Column to partition on
      self.partition_on_column='City_Category'
      
      # Whether the task is 'regression' or 'classification'
      self.task='regression'
  
   def preprocess(self,df):
      # Distracting columns - exclude 'City_Category' as we will partition on it
      cols_to_drop=['User_ID','Product_ID']
      
      # Columns to one-hot-code encode
      one_hot_cols=['Gender','Age','Occupation','Stay_In_Current_City_Years', 'Marital_Status',
                     'Product_Category_1','Product_Category_2', 'Product_Category_3']

      df['Product_Category_2']=df['Product_Category_2'].fillna(0)
      df['Product_Category_3']=df['Product_Category_3'].fillna(0)
      
      # Drop distracting columns
      df=df.drop(columns=cols_to_drop)

      # One-hot-code encodings
      df=pd.get_dummies(df,columns=one_hot_cols)

      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True) 

      return df
   
   
class FlightPriceDataset(Dataset):
   """
   Source: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction
   """

   def __init__(self,fname=f'{DATA_PATH}/flight_price.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='price'

      # Column to partition on
      self.partition_on_column='airline'
      
      # Whether the task is 'regression' or 'classification'
      self.task='regression'
  
   def preprocess(self,df):
      # Distracting columns
      cols_to_drop=['Unnamed: 0','flight']
      
      # Columns to one-hot-code encode
      one_hot_cols=['source_city', 'departure_time', 'arrival_time', 'destination_city']

      # Some custom encoding
      df["stops"] = df["stops"].replace({'zero':0,'one':1,'two_or_more':2}).astype(int)
      df["class"] = df["class"].replace({'Economy':0,'Business':1}).astype(int)

      
      # Drop distracting columns
      df=df.drop(columns=cols_to_drop)

      # One-hot-code encodings
      df=pd.get_dummies(df,columns=one_hot_cols)

      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)

      return df
   
class VehicleLoanDefaultDataset(Dataset):
   """
   Source: https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction
   """

   def __init__(self,fname=f'{DATA_PATH}/car_loan_default.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='loan_default'

      # Column to partition on
      self.partition_on_column='branch_id'
      
      # Whether the task is 'regression' or 'classification'
      self.task='classification'
  
   def preprocess(self,df):

      # Distracting columns
      cols_to_drop=[
         'UniqueID','supplier_id', 'Current_pincode_ID','State_ID','Employee_code_ID',
         'MobileNo_Avl_Flag','PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS','PRI.CURRENT.BALANCE',
         'PRI.SANCTIONED.AMOUNT','SEC.NO.OF.ACCTS','PRI.NO.OF.ACCTS','PRI.DISBURSED.AMOUNT','PRI.ACTIVE.ACCTS', 
         'PRI.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT', 'SEC.OVERDUE.ACCTS',
         'SEC.DISBURSED.AMOUNT','PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT','disbursed_amount','SEC.ACTIVE.ACCTS'
      ]
      df=df.drop(columns=cols_to_drop)


      # Convert duration string to number of years
      def duration_in_years(duration_str):
         years = int(duration_str.split(' ')[0].replace('yrs',''))
         months = int(duration_str.split(' ')[1].replace('mon',''))
         return years+(months/12)
      
      df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(duration_in_years)
      df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(duration_in_years)

      # Calculate an age column
      def birth_year(date_str):
         year = int(date_str.split('-')[-1])
         return year+2000 if year<=25 else year+1900
      
      df['Date.of.Birth'] = df['Date.of.Birth'].apply(birth_year)
      df['DisbursalDate'] = df['DisbursalDate'].apply(birth_year)
      df['Age']=df['DisbursalDate']-df['Date.of.Birth']
      df=df.drop(columns=['Date.of.Birth','DisbursalDate'])
      
      # Columns to one-hot-code encode
      one_hot_cols=['Employment.Type','PERFORM_CNS.SCORE.DESCRIPTION']
      df=pd.get_dummies(df,columns=one_hot_cols)

      # Make explicit the number of features each customer is missing
      df['Missing Features'] = (df == 0).astype(int).sum(axis=1)

      t=df['branch_id'].value_counts()[(df['branch_id'].value_counts()>5000)].index.to_list()
      df=df.loc[df['branch_id'].isin(t)]
      df['branch_id']=df['branch_id'].astype('category').cat.codes

      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)

      return df

   
# DATASETS WITHOUT NATURAL PARTITIONS
   
class CreditCardDefaultDataset(Dataset):
   """
   Source: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
   """

   def __init__(self,fname=f'{DATA_PATH}/credit_card_default.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='default.payment.next.month'
      
      # Whether the task is 'regression' or 'classification'
      self.task='classification'
  
   def preprocess(self,df):
      # Distracting columns
      cols_to_drop=['ID']
      
      # Columns to one-hot-code encode
      one_hot_cols=[]

      # Drop distracting columns
      df=df.drop(columns=cols_to_drop)

      # One-hot-code encodings
      df=pd.get_dummies(df,columns=one_hot_cols)

      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)

      return df
   
   
class CoverTypeDataset(Dataset):
   """
   Source: https://archive.ics.uci.edu/ml/datasets/covertype
   """

   def __init__(self,fname=f'{DATA_PATH}/covtype.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='Cover_Type'
      
      # Whether the task is 'regression' or 'classification'
      self.task='classification'
   
   def preprocess(self,df):

      # Limit the datasets to 2 most common labels
      t=df['Cover_Type'].value_counts()[(df['Cover_Type'].value_counts()>200000)].index.to_list()
      df=df.loc[df['Cover_Type'].isin(t)]

      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)

      return df
   

class DiamondsDataset(Dataset):
   """
   Source: https://www.openml.org/search?type=data&sort=runs&id=42225&status=active
   """

   def __init__(self,fname=f'{DATA_PATH}/diamonds.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='Price'
      
      # Whether the task is 'regression' or 'classification'
      self.task='regression'

  
   def preprocess(self,df):

      # Columns to one-hot-code encode
      one_hot_cols=['Cut','Color','Clarity']
      df=pd.get_dummies(df,columns=one_hot_cols)


      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)

      return df
   
   
class FriedDataset(Dataset):
   """
   Source: https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html
   Note: maybe reduce N?
   """

   def __init__(self,fname=f'{DATA_PATH}/fried_delve.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='y'
      
      # Whether the task is 'regression' or 'classification'
      self.task='regression'
   
   def load_raw(self):
      return pd.read_csv(self.fpath,delimiter=' ')
  
   def preprocess(self,df):
      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)

      return df
   
   
class HotelReservationsDataset(Dataset):
   """
   Source: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset
   """

   def __init__(self,fname=f'{DATA_PATH}/hotel_reservations.csv'):
      # Path to file with data
      self.fpath=fname

      # Column to predict
      self.y_column='booking_status'
      
      # Whether the task is 'regression' or 'classification'
      self.task='classification'
   
  
   def preprocess(self,df):
      # Drop distracting columns
      cols_to_drop=['Booking_ID']
      df=df.drop(columns=cols_to_drop)

      # One-hot encode columns
      one_hot_cols=['type_of_meal_plan','room_type_reserved','market_segment_type']
      df=pd.get_dummies(df,columns=one_hot_cols)

      
      # Reset index and shuffle
      df=df.reset_index(drop=True)
      df=df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)

      return df



def check_partition(partition,df):
   """
   Sanity checks a generated partition against the DataFrame
   it was generated from. 
   """
   # Check they have the same number of elements
   n=sum(y.shape[0] for (_,y) in partition.values())
   assert n==df.shape[0],f'Partition adds up to {n} elements while df to {df.shape[0]}'

   # Check the indices are the same uppon joining
   X,_=utils.join_partitions(partition,partition.keys())
   a=X.index.to_numpy()
   a.sort()
   b=df.index.to_numpy()
   b.sort()
   assert all(a==b),f'Joined index and df\'s are not equal'





    