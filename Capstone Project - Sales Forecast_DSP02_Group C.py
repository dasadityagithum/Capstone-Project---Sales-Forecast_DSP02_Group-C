#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
warnings.filterwarnings("ignore")
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_excel('C:\Deepak\Imarticus\Capstone Project\Sales_Forecast_Data1.xlsx')


# In[3]:


df.isnull().sum()


# In[4]:


df.head()


# In[5]:


df['FG'].value_counts()


# In[6]:


df['COMPANY'].value_counts()


# # Processing of Data

# In[7]:


import statistics


# In[8]:


df["VALUE"].mean()


# In[9]:


df2 = df.groupby([df['COMPANY'],df['FIN_YEAR'],df['FG'],df['MONTH']]).mean()


# In[10]:


print(df2)


# In[11]:


df1 = df[df["VALUE"]!=0]


# In[12]:


df1 = df[df["VALUE"]!=0].mean()


# In[13]:


print(df1)


# In[14]:


df1 = df[df["VALUE"]!=0].groupby([df['COMPANY'],df['FIN_YEAR'],df['FG'],df['MONTH']]).mean()


# In[15]:


df1 = df1.reset_index()


# In[16]:


print(df1)


# In[17]:


# df['VALUE'] = df1['VALUE'].replace(0, df[df["VALUE"]!=0].groupby([df['COMPANY'],df['FIN_YEAR'],df['FG'],df['MONTH']]).mean()) 


# In[18]:


df1.isnull().sum()


# In[19]:


# print(df)


# # Creating the dataframe for ABC Manufacturing

# In[20]:


df1= df[df['COMPANY']==('ABC Manufacturing')]


# In[21]:


df1.AVG=(df1['VALUE'].loc[df1['VALUE']!=0]).mean()


# In[22]:


print(df1.AVG)


# In[23]:


# df1 = df1[df1["VALUE"]!=0].groupby([df1['FIN_YEAR'],df1['STATE'],df1['FG'],df1['MONTH']]).mean()


# In[24]:


# print(df1)


# In[25]:


df1.head()


# In[26]:


# We have 4 years of data


# In[27]:


df1.shape


# In[28]:


# Adding Rainfall Data from 2014-2017 obtained from external source for better analysis


# In[29]:


rain = pd.read_excel('C:\Deepak\Imarticus\Capstone Project\Rainfall_Data_Project.xlsx')


# In[30]:


rain.head(15)


# In[31]:


rain['MONTH']= rain['MONTH'].str.title()


# In[32]:


rain


# In[33]:


rain['YEAR-MONTH']= rain['MONTH'].map(str)+ rain['YEAR'].map(str)


# In[34]:


rain


# In[35]:


# Adding Population and Area data for states for better analysis


# In[36]:


popln = pd.read_excel('C:\Deepak\Imarticus\Capstone Project\Population & Area.xlsx')


# In[37]:


popln


# In[38]:


lst_months=['Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df1 = df1[df1["MONTH"].isin(lst_months)]


# In[39]:


df1.head()


# In[40]:


df1['Year-Month']= df['MONTH'] + df['FIN_YEAR'].str[:4]


# In[41]:


df2=df[df['COMPANY'] == 'ABC Manufacturing']


# In[42]:


lst_months_sec=['Jan','Feb','Mar']
df2 = df2[df2["MONTH"].isin(lst_months_sec)]


# In[43]:


df2['Year-Month']= df2['MONTH'] + df2['FIN_YEAR'].str[5:]


# In[44]:


new_df = pd.concat([df1, df2])


# In[45]:


new_df


# In[46]:


# Merge original data, rainfall and Population-Area based on State and Year-Month


# In[47]:


merg_df = pd.merge(new_df,rain,how= 'inner',left_on=['STATE','Year-Month'],right_on=['SUBDIVISION','YEAR-MONTH'])


# In[48]:


merg_df.head()


# In[49]:


merg_df = pd.merge(merg_df,popln, how='inner', left_on= ['STATE'], right_on=['State or union territory'])


# In[50]:


merg_df['Year-Month']= pd.to_datetime(merg_df['Year-Month'],infer_datetime_format=True)


# In[51]:


merg_df


# In[52]:


merg_df['Population'].value_counts()


# In[53]:


merg_df.info()


# In[54]:


# Finalized data for analysis after merging sales data, rainfall, population & area


# # Data Visualization

# In[55]:


# Company-wise analysis


# In[56]:


sns.countplot(x='STATE', data= df)


# In[57]:


# ABC Company performance analysis


# In[58]:


sns.countplot(x='STATE', data= merg_df)


# In[59]:


merg_df['STATE'].value_counts(dropna= False)


# In[60]:


sns.barplot(x='STATE',y='VALUE',data= merg_df)


# In[61]:


# Uttar Pradesh's stands top in sales/consumption, 2nd is Punjab, 3rd Uttarakhand, 4th Haryana & bottom Himachal Pradesh is higher


# In[62]:


sns.countplot(x='STATE', hue='FG', data = merg_df, palette="mako_r")


# In[63]:


# Pesticides are equally distributed in every state based on their consumption


# In[64]:


sns.barplot(x='FG',y='VALUE',data= merg_df)


# In[65]:


# ABC Manufacturing company aren't supplying Bactericides whereas Insecticides tops the Sales


# In[66]:


sns.barplot(x='STATE',y = 'Rainfall', data= merg_df)


# In[67]:


# Uttarakhand & Himachal Pradesh has highest rainfall v/s low consumption. Where as Uttar Pradesh & Punjab has low rainfal with highest consumption


# In[68]:


sns.barplot(x='STATE', y= 'Area', hue='Population',data= merg_df)


# In[69]:


UP_df= merg_df[(merg_df['STATE']=='Uttar Pradesh')]
ax = UP_df.plot('YEAR-MONTH','VALUE')
ax1 = ax.twinx()
UP_df.plot('YEAR-MONTH','Rainfall',ax=ax1, color='r',kind = 'line')


# In[70]:


HR_df= merg_df[(merg_df['STATE']=='Haryana')]
ax = HR_df.plot('YEAR-MONTH','VALUE')
ax1 = ax.twinx()
HR_df.plot('YEAR-MONTH','Rainfall',ax=ax1, color='r',kind = 'line')


# In[71]:


HP_df= merg_df[(merg_df['STATE']=='Himachal Pradesh')]
ax = HP_df.plot('YEAR-MONTH','VALUE')
ax1 = ax.twinx()
HP_df.plot('YEAR-MONTH','Rainfall',ax=ax1, color='r',kind = 'line')


# In[72]:


PJ_df= merg_df[(merg_df['STATE']=='Punjab')]
ax = PJ_df.plot('YEAR-MONTH','VALUE')
ax1 = ax.twinx()
PJ_df.plot('YEAR-MONTH','Rainfall',ax=ax1, color='r',kind = 'line')


# In[73]:


UK_df= merg_df[(merg_df['STATE']=='Uttarakhand')]
ax = UK_df.plot('YEAR-MONTH','VALUE')
ax1 = ax.twinx()
UK_df.plot('YEAR-MONTH','Rainfall',ax=ax1, color='r',kind = 'line')


# In[74]:


# sns.pairplot(merg_df,hue='VALUE')


# In[75]:


merg_df.info()


# In[76]:


sns.distplot(merg_df['VALUE'],kde=True,color = 'blue',bins=40)


# In[77]:


# We observe that the VALUE which is the target variable is Right Skewed


# In[78]:


g = plt.subplots(figsize=(12, 4))
ax = sns.boxplot(x=merg_df['VALUE'],whis=2)


# In[79]:


merg_df['Year-Month'].value_counts()


# In[80]:


merg_df.isnull().sum()


# In[81]:


merg_df.corr()


# In[82]:


# Rainfall has a low negative correlation with absolute value close to 0 with the Target variable. So, we can drop the rainfall data.


# In[83]:


sns.heatmap(merg_df.corr(),annot= True,cmap="YlOrRd")


# In[84]:


HR_rain =merg_df[merg_df['STATE'] == 'Haryana']


# In[85]:


HR_rain.info()


# In[86]:


HR_rain.corr()


# In[87]:


sns.heatmap(HR_rain.corr(),annot= True,cmap="YlGnBu")


# In[88]:


UP_rain= merg_df[merg_df['STATE']==('Uttar Pradesh')]

PJ_rain= merg_df[merg_df['STATE']==('Punjab')]

UK_rain= merg_df[merg_df['STATE']==('Uttarakhand')]

HP_rain= merg_df[merg_df['STATE']==('Himachal Pradesh')]


# In[89]:


sns.heatmap(UP_rain.corr(), annot =True, cmap='YlGnBu')


# In[90]:


sns.heatmap(PJ_rain.corr(), annot =True, cmap='YlGnBu')


# In[91]:


sns.heatmap(UK_rain.corr(), annot =True, cmap='YlGnBu')


# In[92]:


sns.heatmap(HP_rain.corr(), annot =True, cmap='YlGnBu')


# In[93]:


new_df['Year-Month']= pd.to_datetime(new_df['Year-Month'],infer_datetime_format=True)


# # Checking if the data is Stationary - using Augmneted Dickey Fuller Test

# In[94]:


X= new_df.VALUE
result = adfuller(X)


# In[95]:


print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[96]:


# The p-value is very less than the significance level of 0.05. Hence we can reject the null hypothesis and take that the series is stationary


# In[97]:


new_df = new_df.sort_values('Year-Month')


# In[98]:


new_df


# In[99]:


new_df.drop(['FIN_YEAR','MONTH','FG','DISTRICT','COMPANY'],axis=1,inplace =True)


# In[100]:


new_df


# # Segmentation based on States

# In[101]:


HR=new_df[new_df['STATE'] == 'Haryana']

UP= new_df[new_df['STATE']==('Uttar Pradesh')]

PJ= new_df[new_df['STATE']==('Punjab')]

UK= new_df[new_df['STATE']==('Uttarakhand')]

HP= new_df[new_df['STATE']==('Himachal Pradesh')]


# In[102]:


HR


# In[103]:


UP


# In[104]:


new_df=new_df.groupby('Year-Month')['VALUE'].sum().reset_index()


# In[105]:


HR=HR.groupby('Year-Month')['VALUE'].sum().reset_index()

UP=UP.groupby('Year-Month')['VALUE'].sum().reset_index()

PJ=PJ.groupby('Year-Month')['VALUE'].sum().reset_index()

UK=UK.groupby('Year-Month')['VALUE'].sum().reset_index()

HP=HP.groupby('Year-Month')['VALUE'].sum().reset_index()


# In[106]:


new_df=  new_df.set_index('Year-Month')


# In[107]:


new_df


# In[108]:


HR=  HR.set_index('Year-Month')

UP=  UP.set_index('Year-Month')

PJ=  PJ.set_index('Year-Month')

UK=  UK.set_index('Year-Month')

HP=  HP.set_index('Year-Month')


# In[109]:


HR


# In[110]:


UP


# # Resampling the data based on MS- Monthly Start

# In[111]:


All= new_df['VALUE'].resample('MS').mean()

y= UP['VALUE'].resample('MS').mean()

y1= HR['VALUE'].resample('MS').mean()

y2= PJ['VALUE'].resample('MS').mean()

y3= UK['VALUE'].resample('MS').mean()

y4= HP['VALUE'].resample('MS').mean()


# # Visualizing Pesticides Sales

# In[112]:


All.plot(figsize=(10,8))
plt.title('Sales Forecast of All States')
plt.show()


# In[113]:


y.plot(figsize=(10,8))
plt.title('Sales Forecast of UP')
plt.show()


# In[114]:


y1.plot(figsize=(10,8))
plt.title('Sales Forecast of HR')
plt.show()


# In[115]:


y2.plot(figsize=(10,8))
plt.title('Sales Forecast of PJ')
plt.show()


# In[116]:


y3.plot(figsize=(10,8))
plt.title('Sales Forecast of UK')
plt.show()


# In[117]:


y4.plot(figsize=(10,8))
plt.title('Sales Forecast of HP')
plt.show()


# # Appears that there are distinguishable patterns when we plot the data. The time-series has seasonality pattern.
# 
# # Checking our data with a method called Decompostion. It allows us to decompose the time series data into three distinct components trend, seasonality and noise

# In[118]:


from pylab import rcParams
import statsmodels.api as sm

rcParams['figure.figsize'] =12,8
decomposition = sm.tsa.seasonal_decompose(All,model= 'additive')
fig= decomposition.plot()
plt.show()


# In[119]:


from pylab import rcParams
import statsmodels.api as sm

rcParams['figure.figsize'] =12,8
decomposition = sm.tsa.seasonal_decompose(y,model= 'additive')
fig= decomposition.plot()
plt.show()


# In[120]:


rcParams['figure.figsize'] =12,8
decomposition = sm.tsa.seasonal_decompose(y1,model= 'additive')
fig= decomposition.plot()
plt.show()


# In[121]:


rcParams['figure.figsize'] =12,8
decomposition = sm.tsa.seasonal_decompose(y2,model= 'additive')
fig= decomposition.plot()
plt.show()


# In[122]:


rcParams['figure.figsize'] =12,8
decomposition = sm.tsa.seasonal_decompose(y3,model= 'additive')
fig= decomposition.plot()
plt.show()


# In[123]:


rcParams['figure.figsize'] =12,8
decomposition = sm.tsa.seasonal_decompose(y4,model= 'additive')
fig= decomposition.plot()
plt.show()


# In[124]:


# The plots above clearly shows that the sales are uncertain, along with its seasonality


# # Split the Train and Test Data based on States

# In[125]:


train_len = 38
train_All = All[0:train_len]
test_All = All[train_len:]


# In[126]:


train_len = 38
train = y[0:train_len]
test = y[train_len:]


# In[127]:


train_len1 = 38
train1 = y1[0:train_len1]
test1 = y1[train_len1:]


# In[128]:


train_len2 = 38
train2 = y2[0:train_len2]
test2 = y2[train_len2:]


# In[129]:


train_len3 = 38
train3 = y3[0:train_len3]
test3 = y3[train_len3:]


# In[130]:


train_len4 = 38
train4 = y4[0:train_len4]
test4 = y4[train_len4:]


# In[131]:


train_All


# In[132]:


train


# In[133]:


test_All


# In[134]:


train.plot(figsize=(10,10), title='Average Sales')
test.plot(figsize=(10,10), title='Average Sales')


# In[135]:


train_All.plot(figsize=(10,10), title='Average Sales')
test_All.plot(figsize=(10,10), title='Average Sales')


# In[136]:


train1.plot(figsize=(10,10), title='Average Sales')
test1.plot(figsize=(10,10), title='Average Sales')


# # MODEL BUILDING

# # Holt Winters' additive method with trend and seasonality

# In[137]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
y_hat_hwa = y.copy()


# In[138]:


model = ExponentialSmoothing(np.asarray(y) ,seasonal_periods=12 ,trend='add', seasonal='add')


# In[139]:


model_fit = model.fit(optimized=True)
print(model_fit.params)


# In[140]:


y_hat_pred = model_fit.forecast(46)


# In[141]:


prediction_values=pd.DataFrame(y_hat_pred)


# In[142]:


prediction_values.to_excel("C:\Deepak\Imarticus\Capstone Project\predictions_holds_model.xlsx")


# In[143]:


plt.figure(figsize=(12,4))
plt.plot(y.index,y, label='Observed')
plt.plot(y_hat_hwa.index,y_hat_pred, label='Holt Winters\'s additive forecast')
plt.title('Holt Winters\' Additive Method')
plt.legend(loc='best')
plt.show()


# In[144]:


rmse = np.sqrt(mean_squared_error(y_hat_hwa, y_hat_pred)).round(2)


# In[145]:


print(rmse)


# In[146]:


mape = np.round(np.mean(np.abs(y_hat_hwa-y_hat_pred)/y_hat_hwa)*100,2)


# In[147]:


print(mape)


# In[148]:


from statsmodels.tsa.arima_model import ARIMA


# In[149]:


y.shape


# In[150]:


model = ARIMA(y, order=(0,1,1))


# In[151]:


m1 = model.fit()


# In[152]:


print(m1.summary())


# In[153]:


y_hat= y.copy()
y_pred_AR = m1.predict()


# In[154]:


y_hat[1:].shape


# In[155]:


y_pred_AR.shape


# In[156]:


y_pred_AR


# In[157]:


print(y_hat)


# In[158]:


plt.figure(figsize=(12,4))
plt.plot(y, label='Observed')
plt.plot(y_pred_AR, label='ARIMA forecast')
plt.legend(loc='best')
plt.title('ARIMA Method')
plt.show()


# In[159]:


rmse = np.sqrt(mean_squared_error(y_hat[1:], y_pred_AR)).round(2)


# In[160]:


print(rmse)


# In[161]:


mape = np.round(np.mean(np.abs(y_hat[1:]-y_pred_AR)/y_hat[1:])*100,2)


# In[162]:


print(mape)


# # ARIMA didn't work as there is a seasonality involved. We will try with SARIMA

# # Time series forecasting with SARIMA

# In[163]:


import itertools


# In[164]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[165]:


# Using a “grid search” to find the optimal set of parameters that yields the best performance for our model


# In[166]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(All,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[167]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[168]:


# The above output suggests that ARIMA(0, 1, 1)x(0, 1, 1, 12) yields the lowest AIC value. Therefore we should consider this to be optimal option.


# In[169]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod1 = sm.tsa.statespace.SARIMAX(y1,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results1 = mod1.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results1.aic))
        except:
            continue


# In[170]:


# The above output suggests that ARIMA(0, 1, 1)x(0, 1, 1, 12) yields the lowest AIC value. Therefore we should consider this to be optimal option.


# In[171]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod2 = sm.tsa.statespace.SARIMAX(y2,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results2 = mod2.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results2.aic))
        except:
            continue


# In[172]:


# The above output suggests that ARIMA(0, 1, 1)x(0, 1, 1, 12) yields the lowest AIC value. Therefore we should consider this to be optimal option.


# In[173]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod3 = sm.tsa.statespace.SARIMAX(y3,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results3 = mod3.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results3.aic))
        except:
            continue


# In[174]:


# The above output suggests that ARIMA(0, 1, 1)x(0, 1, 1, 12) yields the lowest AIC value. Therefore we should consider this to be optimal option.


# In[175]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod4 = sm.tsa.statespace.SARIMAX(y4,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results4 = mod4.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results4.aic))
        except:
            continue


# In[176]:


# The above output suggests that ARIMA(0, 1, 1)x(0, 1, 1, 12) yields the lowest AIC value. Therefore we should consider this to be optimal option.


# # Fitting the SARIMA model

# In[177]:


mod = sm.tsa.statespace.SARIMAX(All,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results_All = mod.fit()
print(results_All.summary())


# In[178]:


results_All.plot_diagnostics(figsize=(14, 8))
plt.show()


# In[179]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())


# In[180]:


results.plot_diagnostics(figsize=(14, 8))
plt.show()


# In[181]:


# The Normal Q-Q plot shows that the ordered distribution of residuals follows the distribution similar to normal distribution.


# In[182]:


mod1 = sm.tsa.statespace.SARIMAX(y1,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results1 = mod1.fit()
print(results1.summary())


# In[183]:


results1.plot_diagnostics(figsize=(14, 8))
plt.show()


# In[184]:


mod2 = sm.tsa.statespace.SARIMAX(y2,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results2 = mod2.fit()
print(results2.summary())


# In[185]:


results2.plot_diagnostics(figsize=(14, 8))
plt.show()


# In[186]:


mod3 = sm.tsa.statespace.SARIMAX(y3,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results3 = mod3.fit()
print(results3.summary())


# In[187]:


results3.plot_diagnostics(figsize=(14, 8))
plt.show()


# In[188]:


mod4 = sm.tsa.statespace.SARIMAX(y4,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results4 = mod4.fit()
print(results4.summary())


# In[189]:


results4.plot_diagnostics(figsize=(14, 8))
plt.show()


# # Validating forecasts
# 
# # To check the accuracy of our forecasts, we compare predicted sales to real sales of the time series, and we set forecasts to start at 2017–01–01 to the end of the data.

# In[190]:


pred_All = results_All.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci_All = pred_All.conf_int()
ax = All['2014':].plot(label='observed')
pred_All.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=10, figsize=(14, 7))
ax.fill_between(pred_ci_All.index,
                pred_ci_All.iloc[:, 0],
                pred_ci_All.iloc[:, 1], color='k', alpha=.20)
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[191]:


All_forecasted = pred_All.predicted_mean
All_truth = All['2017-01-01':]
mse = ((All_forecasted - All_truth)**2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[192]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 3)))


# In[193]:


mape_s_All = np.round(np.mean(np.abs(All_truth-All_forecasted)/All_truth)*100,2)


# In[194]:


print(mape_s_All)


# In[195]:


pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=10, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.20)
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[196]:


y_forecasted = pred.predicted_mean
y_truth = y['2016-07-01':]
mse = ((y_forecasted - y_truth)**2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[197]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 3)))


# In[198]:


mape_s = np.round(np.mean(np.abs(y_truth-y_forecasted)/y_truth)*100,2)


# In[199]:


print(mape_s)


# In[200]:


pred1 = results1.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci1 = pred1.conf_int()
ax = y1['2014':].plot(label='observed')
pred1.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci1.index,
                pred_ci1.iloc[:, 0],
                pred_ci1.iloc[:, 1], color='k', alpha=.20)
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[201]:


y_forecasted1 = pred1.predicted_mean
y_truth1 = y1['2017-01-01':]
mse1 = ((y_forecasted1 - y_truth1)**2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse1, 2)))


# In[202]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse1), 3)))


# In[203]:


mape_s1 = np.round(np.mean(np.abs(y_truth1-y_forecasted1)/y_truth1)*100,2)
print(mape_s1)


# In[204]:


pred2 = results2.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci2 = pred.conf_int()
ax = y2['2014':].plot(label='observed')
pred2.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci2.index,
                pred_ci2.iloc[:, 0],
                pred_ci2.iloc[:, 1], color='k', alpha=.20)
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[205]:


y_forecasted2 = pred2.predicted_mean
y_truth2 = y2['2017-01-01':]
mse2 = ((y_forecasted2 - y_truth2)**2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse2, 2)))


# In[206]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse2), 3)))


# In[207]:


mape_s2 = np.round(np.mean(np.abs(y_truth2-y_forecasted2)/y_truth2)*100,2)
print(mape_s2)


# In[208]:


pred3 = results3.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci3 = pred.conf_int()
ax = y3['2014':].plot(label='observed')
pred3.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci3.index,
                pred_ci3.iloc[:, 0],
                pred_ci3.iloc[:, 1], color='k', alpha=.50)
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[209]:


y_forecasted3 = pred3.predicted_mean
y_truth3 = y3['2017-01-01':]
mse3 = ((y_forecasted3 - y_truth3)**2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse3, 2)))


# In[210]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse3), 3)))


# In[211]:


mape_s3 = np.round(np.mean(np.abs(y_truth3-y_forecasted3)/y_truth3)*100,2)


# In[212]:


print(mape_s3)


# In[213]:


pred4 = results4.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci4 = pred4.conf_int()
ax = y4['2014':].plot(label='observed')
pred4.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci4.index,
                pred_ci4.iloc[:, 0],
                pred_ci4.iloc[:, 1], color='k', alpha=.20)
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[214]:


y_forecasted4 = pred4.predicted_mean
y_truth4 = y4['2017-01-01':]
mse4 = ((y_forecasted4 - y_truth4)**2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse4, 2)))


# In[215]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse4), 3)))


# In[216]:


mape_s4 = np.round(np.mean(np.abs(y_truth4-y_forecasted4)/y_truth4)*100,2)
print(mape_s4)


# # Prediction for next 12 months

# In[217]:


pred_uc_All = results_All.get_forecast(steps=12)
pred_ci_All = pred_uc_All.conf_int()
ax = All.plot(label='observed',figsize=(14, 7))
pred_uc_All.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_All.index,
                pred_ci_All.iloc[:, 0],
                pred_ci_All.iloc[:, 1], color='k', alpha=.25,ls='--')
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[218]:


pred_uc = results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed',figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25,ls='--')
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[219]:


pred_uc1 = results1.get_forecast(steps=12)
pred_ci1 = pred_uc1.conf_int()
ax = y1.plot(label='observed',figsize=(14, 7))
pred_uc1.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci1.index,
                pred_ci1.iloc[:, 0],
                pred_ci1.iloc[:, 1], color='k', alpha=.25,ls='--')
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[220]:


pred_uc2 = results2.get_forecast(steps=12)
pred_ci2 = pred_uc2.conf_int()
ax = y2.plot(label='observed',figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci2.index,
                pred_ci2.iloc[:, 0],
                pred_ci2.iloc[:, 1], color='k', alpha=.25,ls='--')
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[221]:


pred_uc3 = results3.get_forecast(steps=12)
pred_ci3 = pred_uc3.conf_int()
ax = y3.plot(label='observed',figsize=(14, 7))
pred_uc3.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci3.index,
                pred_ci3.iloc[:, 0],
                pred_ci3.iloc[:, 1], color='k', alpha=.25,ls='--')
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[222]:


pred_uc4 = results4.get_forecast(steps=12)
pred_ci4 = pred_uc4.conf_int()
ax = y4.plot(label='observed',figsize=(14, 7))
pred_uc4.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci4.index,
                pred_ci4.iloc[:, 0],
                pred_ci4.iloc[:, 1], color='k', alpha=.25,ls='--')
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[223]:


# The line plot is showing the observed values compared to the rolling forecast predictions. Overall, our forecasts align with the true values very well


# In[224]:


# print(pred_uc4 = results4.get_forecast(steps=12))


# In[225]:


# pd.pred_uc4('display.max_rows', 10)


# # visualizing forecasts

# In[226]:


pred_uc_All = results_All.get_forecast(steps=100)
pred_ci_All = pred_uc_All.conf_int()
ax = All.plot(label='observed',figsize=(14, 7))
pred_uc_All.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_All.index,
                pred_ci_All.iloc[:, 0],
                pred_ci_All.iloc[:, 1], color='k', alpha=.25,ls='--')
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[227]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Pesticide Sales')
plt.legend()
plt.show()


# In[228]:


# There is an upward trend in sales in the coming years.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




