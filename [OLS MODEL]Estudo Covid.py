# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:05:40 2020

@author: vitin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


### IMPORTANDO DADOS
covid = pd.read_csv('covid.csv', sep=';')
bolso = pd.read_csv('bolsonaro1.csv')

### MANIPULANDO VARIÁVEL EXPLICATIVA
covid_sp = covid[1558:1639]
covid_sp_casos = covid_sp['casosAcumulados']
covid_sp_novos = covid_sp['casosNovos']
x = covid_sp_casos[1:]
x2 = np.array(covid_sp_novos[1:])

### MANIPULANDO VARIÁVEL INDEPENDENTE
bolso_x = bolso[:][1:]
y = bolso_x.astype(int)


### ESTIMANDO REGRESSÃO
est = sm.OLS(bolso_x.astype(int), x2).fit()
print(est.summary())

### PLOTANDO GRÁFICOS
plt.scatter(bolso_x.astype(int), x2)
sns.regplot(y, x2)