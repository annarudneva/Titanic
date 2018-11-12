#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('train.csv', ',')
pvt = df.pivot_table('PassengerId', 'Pclass', 'Survived', 'count')
pvt.plot(kind = 'bar')
plt.show()


# In[3]:


fig, axes = plt.subplots(ncols = 2)
df.pivot_table('PassengerId', 'SibSp', 'Survived', 'count').plot(ax = axes[0])
df.pivot_table('PassengerId', 'Parch', 'Survived', 'count').plot(ax = axes[1])
plt.show()


# In[4]:


print('Count of row where Age is Null:')
print(df.PassengerId[df.Age.notnull()].count())


# In[5]:


df.Age = df.Age.median()


# In[6]:


df[df.Embarked.isnull()]


# In[7]:


MaxPassEmbarked = df.groupby('Embarked').count()['PassengerId']
df.Embarked[df.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]


# In[8]:


print('Count of row where Fare is Null:')
print(df.PassengerId[df.Fare.isnull()].count())


# In[9]:


df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)


# In[10]:


print('Количество пассажиров мужского пола из 1 класса:')
print(df.Pclass[df.Sex == 'male'][df.Pclass == 1].count())


# In[11]:


print('Количество детей (младше 14) из 2 класса:')
print(df.Pclass[df.Age <= 14][df.Pclass == 1].count())


# In[12]:


print('Количество одиноких пассажиров:')
print(df.Pclass[df.Parch == 0].count())


# In[13]:


print('количество пассажиров, севших в порту Queenstown с дорогими билетами (цена выше средней):')
print(df.Fare[df.Embarked == 'Q'][df.Fare > df.Fare.median()].count())


# In[14]:


print('Cредний возраст пассажиров женского пола:')
print(df.Age[df.Sex == 'female'].median())


# In[15]:


print('Количество одиноких пожилых людей(старше 65):')
print(df.Age[df.Parch == 0][df.Age >= 65].count())


# In[16]:


print('Статистика по детям (младше 14 лет):')
print(df[df.Age <= 14])


# In[17]:


print('Средняя цена билета по портам:')
print(df.groupby('Embarked')['Fare'].median())


# In[18]:


print('Средняя цена билета для каждого социального класса:')
df.groupby('Pclass')['Fare'].median()


# In[ ]:




