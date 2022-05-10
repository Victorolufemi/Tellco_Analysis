import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class Eda:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def col_assign(self, df: pd.DataFrame):
        num_col, cat_col = [], []
        for i in df.columns:
            if df[i].dtypes == 'float':
                num_col.append(i)
            else:
                cat_col.append(i)

        return num_col, cat_col

    def fix_cat_cols(self, df: pd.DataFrame, cat_col):
        for i in cat_col:
            df[i] = df[i].fillna(df[i].mode()[0])
        return df

    def fix_num_cols(self, df: pd.DataFrame, num_col):
        for i in num_col:
            df[i] = df[i].fillna(df[i].mean())
        return df

    def top_produce(self, df: pd.DataFrame, col, number):
        top_col = df[col].value_counts().nlargest(number).to_frame()
        return top_col

    def aggregation(self, df: pd.DataFrame, number):
        group = []
        new_df = df.loc[df["Handset Manufacturer"] == "Apple", :]
        new_df1 = df.loc[df["Handset Manufacturer"] == "Samsung", :]
        new_df2 = df.loc[df["Handset Manufacturer"] == "Huawei", :]
        num_large = new_df['Handset Type'].value_counts().nlargest(number)
        num_large1 = new_df1['Handset Type'].value_counts().nlargest(number)
        num_large2 = new_df2['Handset Type'].value_counts().nlargest(number)
        num_large = pd.DataFrame(num_large.reset_index())
        num_large1 = pd.DataFrame(num_large1.reset_index())
        num_large2 = pd.DataFrame(num_large2.reset_index())
        num_large.rename(columns={'index': 'Apple', 'Handset Type': 'No_apple_product'}, inplace=True)
        num_large1.rename(columns={'index': 'Samsung', 'Handset Type': 'No_samsung_product'}, inplace=True)
        num_large2.rename(columns={'index': 'Huawei', 'Handset Type': 'No_huawei_product'}, inplace=True)
        group.append(num_large)
        group.append(num_large1)
        group.append(num_large2)
        group = pd.concat(group, axis=1)
        return group

    """# **TASK ONE**
    
    ### 1.1
    """

    def group_count(self, df: pd.DataFrame, x, y):
        x_per_y = pd.DataFrame(df.groupby([x]).agg({y: 'count'}).reset_index())
        return x_per_y

    def group_sum(self, df: pd.DataFrame, x, y):
        x_persum_y = pd.DataFrame(df.groupby([x]).agg({y: 'sum'}).reset_index())
        return x_persum_y

    def group_double_sum(self, df: pd.DataFrame, x, y, z):
        xy_per_z = pd.DataFrame(df.groupby([x, y]).agg({z: 'count'}).reset_index())
        return xy_per_z

    # aggregates_user = []
    # Dur_Per_User = []
    # Total_Data_Vol_user = []
    # Dl_UL_per_user = []
    '''
    def aggregate(datas):
      agg2 = pd.DataFrame(datas.groupby(['Total_Data_Volume(MB)']).agg(TOTAL_DUR = ('MSISDN/Number', sum)).reset_index())
      aggregates_user.append(agg)
      Dur_Per_User.append(agg1)
      Total_Data_Vol_user.append(agg2)
      Dl_UL_per_user.append(agg3)
    aggregate(data)
    aggregates_user = pd.concat(aggregates_user, axis=1)
    Dur_Per_User = pd.concat(Dur_Per_User, axis=1)
    Total_Data_Vol_user = pd.concat(Total_Data_Vol_user, axis=1)
    Dl_UL_per_user = pd.concat(Dl_UL_per_user, axis=1)
    '''
    """### 1.2.3"""

    def convertbyte_scale(self, df: pd.DataFrame, substrings, replaces, div_value):
        my_bytes = [j for j in df.columns if substrings in j]
        for i in my_bytes:
            df[i.replace(substrings, replaces)] = round((df[i] / div_value), 3)
            df.drop(i, axis=1, inplace=True)
        return df

    def min_scale(self, df: pd.DataFrame, col_names, x, y):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(x, y))
        df[col_names] = scaler.fit_transform(df[col_names])
        return df

    def standard_scale(self, df: pd.DataFrame, col_names):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[col_names] = scaler.fit_transform(df[col_names])
        return df

    def describes(self, df: pd.DataFrame, relevant_num):
        for cols in relevant_num:
            print(df[cols].describe().to_frame())
            print(f"Column name is {cols}")

    def non_grahical_EDA(self, df: pd.DataFrame, relevant_num):
        for cols in relevant_num:
            print(f"Column name is {cols}")
            print(f'skewness for this column is {df[cols].skew()}')
            print(f'kurtosis for this column is {df[cols].kurtosis()}')
            Q3, Q1 = np.percentile(df[cols], [75, 25])
            IQR = Q3 - Q1
            print(f'The IQR is {IQR}')
            print(f'The number of Unique value of column {cols} is : {df[cols].nunique()}')
            print('____________________________________________________________________')

    def univariate_plot(self, df: pd.DataFrame, relevant_num):
        for cols in relevant_num:
            sns.histplot(data=df, x=cols)
            plt.show()
        for cols in relevant_num:
            sns.boxplot(data=df, x=cols)
            plt.show()
        for cols in relevant_num:
            sns.kdeplot(data=df, x=cols)
            plt.show()

    def bivariate_plot(self, df: pd.DataFrame, relevant_app, x):
        for i in relevant_app:
            sns.scatterplot(data=df, x=x, y=i, alpha=0.5)
            plt.title(f'graph of {i} against {x}')
            plt.xlabel(x)
            plt.ylabel(i)
            plt.show()

    def variable_transformation(self, df: pd.DataFrame,x,y):
        newl = []
        df[y] = pd.qcut(df[x], 10,labels=False,duplicates= 'drop')
        New_df = pd.DataFrame()
        New_df['MSISDN/Number'] = df['MSISDN/Number']
        New_df['total_data_volume'] = df['Total_data (MB)']
        New_df['top_5_decile_Dur. (s)'] = df[y]
    
        new_df = New_df.loc[New_df["top_5_decile_Dur. (s)"]==3,:]
        new_df1 = New_df.loc[New_df["top_5_decile_Dur. (s)"]==2,:]
        new_df2 = New_df.loc[New_df["top_5_decile_Dur. (s)"]==0,:]
        new_df3 = New_df.loc[New_df["top_5_decile_Dur. (s)"]==6,:]
        new_df4 = New_df.loc[New_df["top_5_decile_Dur. (s)"]==7,:]
    
        new_df = pd.DataFrame(new_df.reset_index())
        new_df1 = pd.DataFrame(new_df1.reset_index())
        new_df2 = pd.DataFrame(new_df2.reset_index())
        new_df3 = pd.DataFrame(new_df3.reset_index())
        new_df4 = pd.DataFrame(new_df4.reset_index())
    
    
    
        newl.append(new_df)
        newl.append(new_df1)
        newl.append(new_df2)
        newl.append(new_df3)
        newl.append(new_df4)
    
        top_5s = pd.concat(newl,axis=0)
    
        top_5s.drop("index",axis=1,inplace=True)
        return top_5s
    
    def corr(self, df: pd.DataFrame, cols):
        df_data = pd.DataFrame()
        for i in cols:
            df_data[i] = df[i]
        return df_data.corr()

    def PCA(self, df: pd.DataFrame, principal1_name, principal2_name):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data=principalComponents, columns=[principal1_name, principal2_name])
        return principalDf

    def fix_outlier(self, df: pd.DataFrame, column):
        for i in column:
            df[i] = np.where(df[i] > df[i].quantile(0.95),
                             df[i].mean(),
                             df[i])
        return df
