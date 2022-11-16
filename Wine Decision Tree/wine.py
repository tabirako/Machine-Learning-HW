#!/usr/bin/env python
# coding: utf-8

# ### 需要的函式

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn import tree
import pydotplus
import graphviz


# In[2]:


wine = load_wine()


# ### data是資料 target是目標的類別

# In[3]:


X = wine.data
y = wine.target


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ### 載入訓練和測試集

# In[5]:


clf = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train,y_train)


# ### 預測成果

# In[17]:


print(clf.score(X_test,y_test))


# In[7]:


#改中文
wine.feature_names = ['酒精',
                      '蘋果酸',
                      '灰',
                      '灰分的鹼度',
                      '鎂',
                      '總酚',
                      '黃酮類化合物',
                      '非黃烷類酚類',
                      '原花色素',
                      '顏色強度',
                      '色調',
                      '稀釋葡萄酒的OD280 / OD315',
                      '脯氨酸']
                      


# ### 印出決策樹

# In[8]:




dot_data = tree.export_graphviz(clf, out_file='tree.dot', feature_names=wine.feature_names, class_names=wine.target_names, filled=True, rounded=True)
with open("tree.dot", encoding='utf-8') as f:
    dot_graph = f.read()
    graph = graphviz.Source(dot_graph.replace("helvetica", "PMingLiu"))
graph.view()


# 

# In[ ]:




