#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt


# In[3]:


import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_cbe136da66dc461989fb771dc3f9946e = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='5v4Grj0PCaAjaU-hlGEuCuN_u3EL77Vll0vJ8GqV36Ag',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_cbe136da66dc461989fb771dc3f9946e.get_object(Bucket='opi-donotdelete-pr-31zvkcjimgz06s',Key='bops_bm.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_bm = pd.read_csv(body)

body = client_cbe136da66dc461989fb771dc3f9946e.get_object(Bucket='opi-donotdelete-pr-31zvkcjimgz06s',Key='bops_online.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_online = pd.read_csv(body)


# In[4]:


df_online.head()


# In[5]:


df_online.columns


# In[6]:


df_online=df_online[['id (DMA)','year','month','week','after','close',' sales ']]
df_online.head()


# In[7]:


df_bm.head()


# In[8]:


df_bm.columns


# In[9]:


df_bm=df_bm[['id (store)', 'year', 'month', 'week', 'usa', 'after', ' sales ']]
df_bm.head()


# In[10]:


pd.DataFrame(df_bm.dtypes,columns=['Type']).T


# ## Missing Values

# In[11]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[12]:


missing_data(df_bm)


# In[13]:


missing_data(df_online)


# In[14]:


df_bm.replace([np.inf, -np.inf], np.nan)
df_bm=df_bm.dropna()


# In[15]:


df_bm[['id (store)', 'year', 'month', 'week', 'usa', 'after']]=df_bm[['id (store)', 'year', 'month', 'week', 'usa', 'after']].astype('int')


# In[16]:


df_bm.head()


# In[17]:


df_bm[' sales '] = df_bm[' sales '].str.replace(',', '').astype(float)


# In[18]:


df_online[' sales '] = df_online[' sales '].str.replace(',', '').astype(float)


# In[19]:


df_bm.head()


# In[20]:


df_online.head()


# In[21]:


df_online.describe()


# In[22]:


df_bm.describe()


# In[23]:


df_bm['date'] = pd.to_datetime(df_bm.year.astype(str), format='%Y') +              pd.to_timedelta(df_bm.week.mul(7).astype(str) + ' days')
df_bm.head()


# In[24]:


df_online['date'] = pd.to_datetime(df_online.year.astype(str), format='%Y') +              pd.to_timedelta(df_online.week.mul(7).astype(str) + ' days')
df_online.head()


# In[25]:


df_online


# ## Graficas ventas online

# In[26]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)
sales_mean=df_online[' sales '].groupby(df_online['date']).mean()
sales_median=df_online[' sales '].groupby(df_online['date']).median()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title(' Sales Online - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# 

# In[27]:


df_online1= df_online[df_online.after == 1]
df_online2= df_online[df_online.after != 1]
df_online2.head()


# In[28]:


sales_mean=df_online2[' sales '].groupby(df_online2['date']).mean()
sales_median=df_online1[' sales '].groupby(df_online1['date']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Before', 'After'], loc='best', fontsize=16)
plt.title(' Sales Online - Mean ', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# Se muestra una clara disminucion de ventas a partir de la implementacion de la estrategia BOPS. Sin embargo, aun faltan mas facotres a considerar como la DMA

# In[29]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)
sales_mean=df_online[' sales '].groupby(df_online['week']).mean()
sales_median=df_online[' sales '].groupby(df_online['week']).median()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title('Weekly Sales Online - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# In[30]:



sales_mean=df_online[' sales '].groupby(df_online['month']).mean()
sales_median=df_online[' sales '].groupby(df_online['month']).median()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title('Monthly Sales Online - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# In[31]:


sales_mean=df_bm[' sales '].groupby(df_bm['date']).mean()
sales_median=df_bm[' sales '].groupby(df_bm['date']).median()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title('Sales bm - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# In[32]:


df_bm1= df_bm[df_bm.after == 1]
df_bm2= df_bm[df_bm.after != 1]
df_bm2.head()


# In[33]:


sales_mean=df_bm2[' sales '].groupby(df_bm2['date']).mean()
sales_median=df_bm1[' sales '].groupby(df_bm1['date']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Before', 'After'], loc='best', fontsize=16)
plt.title(' Sales bm - Mean ', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# Aunque se muestra una disminición en las ventas a partir de la implementación de la estrategia, aun falta considerar la localización de las tiendas

# In[34]:



sales_mean=df_bm[' sales '].groupby(df_bm['week']).mean()
sales_median=df_bm[' sales '].groupby(df_bm['week']).median()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title('Weekly Sales bm - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# In[35]:



sales_mean=df_bm[' sales '].groupby(df_bm['month']).mean()
sales_median=df_bm[' sales '].groupby(df_bm['month']).median()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title('Monthly Sales bm - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# In[36]:


weekly_sales = df_bm[' sales '].groupby(df_bm['id (store)']).mean()
plt.figure(figsize=(20,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales - per Store', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.show()


# In[37]:


weekly_sales = df_online[' sales '].groupby(df_online['id (DMA)']).mean()
plt.figure(figsize=(20,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales - per Store', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.show()


# ## Analizarames los datos a partir de la variable close y la variable usa
# 
# Analizaremos las ventas para las compras realizadas cerca de la DMA y las lejanas por separado. Ademas se analiza las tiendas que se encuentran en Estados Unidos, debido a que solo en Estados Unidos se lanzo el proyecto BOPS

# ### DMA

# In[38]:


df_onlineDMA1b= df_online2[df_online2.close == 1]
df_onlineDMA1a= df_online1[df_online1.close == 1]

df_onlineDMA2b= df_online2[df_online2.close != 1]
df_onlineDMA2a= df_online1[df_online1.close != 1]


# In[39]:


sales_mean=df_onlineDMA1b[' sales '].groupby(df_onlineDMA1b['date']).mean()
sales_median=df_onlineDMA1a[' sales '].groupby(df_onlineDMA1a['date']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Before', 'After'], loc='best', fontsize=16)
plt.title(' Sales Online DMA=1 - Mean ', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# In[40]:


sales_mean=df_onlineDMA2b[' sales '].groupby(df_onlineDMA2b['date']).mean()
sales_median=df_onlineDMA2a[' sales '].groupby(df_onlineDMA2a['date']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Before', 'After'], loc='best', fontsize=16)
plt.title(' Sales Online DMA=0 - Mean ', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# ### Tiendas en Estados Unidos

# In[41]:


df_bmDMA1b= df_bm2[df_bm2.usa == 1]
df_bmDMA1a= df_bm1[df_bm1.usa == 1]

df_bmDMA2b= df_bm2[df_bm2.usa != 1]
df_bmDMA2a= df_bm1[df_bm1.usa != 1]


# In[42]:


sales_mean=df_bmDMA1b[' sales '].groupby(df_bmDMA1b['date']).mean()
sales_median=df_bmDMA1a[' sales '].groupby(df_bmDMA1a['date']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Before', 'After'], loc='best', fontsize=16)
plt.title(' Sales BM USA - Mean ', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# In[43]:


sales_mean=df_bmDMA2b[' sales '].groupby(df_bmDMA2b['date']).mean()
sales_median=df_bmDMA2a[' sales '].groupby(df_bmDMA2a['date']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(sales_mean.index, sales_mean.values)
sns.lineplot(sales_median.index, sales_median.values)
plt.grid()
plt.legend(['Before', 'After'], loc='best', fontsize=16)
plt.title(' Sales Online Canada - Mean ', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()


# In[44]:


sales_mean_dbbUSA=df_bmDMA1b[' sales '].mean()
sales_mean_dbaUSA=df_bmDMA1a[' sales '].mean()
sales_mean_dbbC=df_bmDMA2b[' sales '].mean()
sales_mean_dbaC=df_bmDMA2a[' sales '].mean()

df_salesmdb=pd.DataFrame({'Tiendas':['USA antes','USA despues','Cananda antes','Canada Despues'],'Venta media':[sales_mean_dbbUSA,sales_mean_dbaUSA,sales_mean_dbbC,sales_mean_dbaC]})
df_salesmdb


# In[48]:


sales_total_dbbUSA=df_bmDMA1b[' sales '].sum()
sales_total_dbaUSA=df_bmDMA1a[' sales '].sum()
sales_total_dbbC=df_bmDMA2b[' sales '].sum()
sales_total_dbaC=df_bmDMA2a[' sales '].sum()

df_salesmdb['Venta Total']= [sales_total_dbbUSA,sales_total_dbaUSA,sales_total_dbbC,sales_total_dbaC]
df_salesmdb


# In[46]:


sales_mean_onb1=df_onlineDMA1b[' sales '].mean()
sales_mean_ona1=df_onlineDMA1a[' sales '].mean()
sales_mean_onb0=df_onlineDMA2b[' sales '].mean()
sales_mean_ona0=df_onlineDMA2a[' sales '].mean()

df_salesmon=pd.DataFrame({'Tiendas':['close antes','Close despues','Far antes','Far Despues'],'Venta media':[sales_mean_dbbUSA,sales_mean_dbaUSA,sales_mean_dbbC,sales_mean_dbaC]})
df_salesmon


# In[53]:


sales_total_onb1=df_onlineDMA1b[' sales '].sum()
sales_total_ona1=df_onlineDMA1a[' sales '].sum()
sales_total_onb2=df_onlineDMA2b[' sales '].sum()
sales_total_ona2=df_onlineDMA2a[' sales '].sum()

df_salesmon['Venta Total']= [sales_total_dbbUSA,sales_total_dbaUSA,sales_total_dbbC,sales_total_dbaC]
df_salesmon


# ### 1. ¿Deberían expandirse a Canadá? 
# 
# Al analizar el promedio de ventas separando las tiendas por pais, se puede observar que la disminución en las ventas de las tiendas en Estados Unidos no es tan grande. No podemos decir que la causa de la disminución de las ventas es debido la implementacion de la nueva estrategia, existen otros factores, como la epoca del año, se sabe que los primeros meses del año son los de menor venta.
# 
# En cuanto al analisis de las ventas online, si se puede apreciar que han disminuido significativamente a partir de la implementación de la estrategia. Esto puede deberse a que, la gente prefiere recojer en la tienda y a la vez comprar productos ahi.
# 
# Por otra parte al analizar las graficas de las tiendas de Canada, se observa una disminución mayor de las ventas promedio a pesar de que en esta region no se implemento la nueva estrategia.Con esto podriamos nos podemos dar una de idea de que en la disminución de las ventas han intervenido diferentes factores a la implementación de la nueva estrategia.
# 
# Con el estudio realizado puedo concluir que la estrategia deberia expandirse a Canada debido a que no hay datos significativos que nos permitan demostrar que la baja de ventas fue causada solamente por la implementación de la estrategia. Ademas de que es muy pronto para que la estrategia sea calificada adecuadamente.
# 
# 

# ### 2. ¿Cuántos millones de dólares se ganaron o perdieron a partir delprograma?Explicatu                             razonamiento y metodología. 

# Para analizar este punto, solo se deberian considerar las tiendas en Estados Unidos debido a que solo en ellas se aplico la nueva estrategia. Ademas de considerar las ventas online con un DMA que cuenten con una tienda cerca. Creo que estos dos casos son los mas impactados por la nueva estrategia.
# 
# Explicado lo anterior muestro la diferencia media de veentas, asi como la diferencia del total de ventas.

# In[54]:


perdida= pd.DataFrame({'Casos':['Online DMA cerca','Tiendas USA'],'Venta Media Antes':[sales_mean_dbbUSA,sales_mean_onb1],'Venta media Despues':[sales_mean_dbaUSA,sales_mean_ona1],
                      'Diferencia venta media':[sales_mean_dbbUSA-sales_mean_dbaUSA,sales_mean_onb1-sales_mean_ona1],
                      'Venta total antes':[sales_total_dbbUSA,sales_total_onb1],'Venta total despues':[sales_total_dbaUSA,sales_total_ona1]})


# In[56]:


perdida['Diferencia venta total']=[sales_total_dbbUSA-sales_total_dbaUSA,sales_total_onb1-sales_total_ona1]
perdida


# In[ ]:




