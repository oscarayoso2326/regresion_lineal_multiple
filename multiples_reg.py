
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('multiples reg.txt')

dataset.cliente_cod=dataset.cliente_cod.astype(str)

dataset.programa_tvt=dataset.programa_tvt.replace('C',1)
dataset.programa_tvt=dataset.programa_tvt.replace('T',2)

clientes=dataset[['cliente_cod']].drop_duplicates()

dataset=dataset[['cliente_cod','plan_mes','programa_tvt','pedido_prv_sol']]


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor = LinearRegression()

df2=pd.DataFrame({'cliente_cod':[],
                  'tipo c':[],
                    'tipo t':[]})
   

# hasta 50 prediciones
for i in range(0,50):
    
    
    sub=dataset[dataset['cliente_cod']==clientes.iloc[i,0]]
    
    X = sub.iloc[:, 1:3].values
    y = sub.iloc[:, 3:4].values
    
    regressor.fit(X, y)     
    
    y_pred2=regressor.predict([[202004,1],[202004,2]])
    y_pred2[0] 
    y_pred2[1] 
    df =  pd.DataFrame({'cliente_cod':[clientes.iloc[i,0]]
                    ,'tipo c':y_pred2[0],
                    'tipo t':y_pred2[1]})
    
    df2=df2.append(df)
