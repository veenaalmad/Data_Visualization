#!/usr/bin/env python
# coding: utf-8

# In[12]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load in the dataset into a pandas dataframe
diamonds = pd.read_csv('diamonds.csv')


# In[3]:


# convert cut, color, and clarity into ordered categorical types
ordinal_var_dict = {'cut': ['Fair','Good','Very Good','Premium','Ideal'],
                    'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                    'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']}

for var in ordinal_var_dict:
    pd_ver = pd.__version__.split(".")
    if (int(pd_ver[0]) > 0) or (int(pd_ver[1]) >= 21): # v0.21 or later
        ordered_var = pd.api.types.CategoricalDtype(ordered = True,
                                                    categories = ordinal_var_dict[var])
        diamonds[var] = diamonds[var].astype(ordered_var)
    else: # pre-v0.21
        diamonds[var] = diamonds[var].astype('category', ordered = True,
                                             categories = ordinal_var_dict[var])


# ## Multivariate Exploration
# 
# In the previous workspace, you looked at various bivariate relationships. You saw that the log of price was approximately linearly related to the cube root of carat weight, as analogy to its length, width, and depth. You also saw that there was an unintuitive relationship between price and the categorical quality measures of cut, color, and clarity, that the median price decreased with increasing quality. Investigating the distributions more clearly and looking at the relationship between carat weight with the three categorical variables showed that this was due to carat size tending to be smaller for the diamonds with higher categorical grades.
# 
# The goal of this workspace will be to depict these interaction effects through the use of multivariate plots.
# 
# To start off with, create a plot of the relationship between price, carat, and clarity. In the previous workspace, you saw that clarity had the clearest interactions with price and carat. How clearly does this show up in a multivariate visualization?

# In[6]:


# multivariate plot of price by carat weight, and clarity
col_classes = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
colclasses = pd.api.types.CategoricalDtype(ordered = True, categories = col_classes)
diamonds['color'] = diamonds['color'].astype(colclasses)


# In[7]:


diamonds['carat'] = diamonds[(diamonds['carat'] < 1.05) & (diamonds['carat'] > 0.95)]
plt.figure(figsize=[10,7])
ax = sb.pointplot(data = diamonds, x = 'cut', y = 'price', hue = 'color', palette = 'viridis', dodge = 0.7, linestyles = "")


# Price by Carat and Clarity Comment 1: With two numeric variables and one categorical variable, there are two main plot types that make sense. A scatterplot with points colored by clarity level makes sense on paper, but the sheer number of points causes overplotting that suggests a different plot type. A faceted scatterplot or heat map is a better choice in this case.</span>

# In[25]:


# multivariate plot of price by carat weight, and clarity
g = sb.FacetGrid(data=diamonds, col='clarity',  col_wrap=4)
g.map(plt.scatter, 'carat', 'price');


# In[20]:


def cube_root_trans(x, inverse = False):
    """ transformation helper function """
    if not inverse:
        return x**(1./3)
    else:
        return x**3
diamonds['cr_carat'] = diamonds['carat'].apply(cube_root_trans)
diamonds['lg_price'] = diamonds['price'].apply(np.log10)
g = sb.FacetGrid(data=diamonds, col='clarity', col_wrap=2, size=6)
g.map(plt.scatter, 'cr_carat', 'lg_price');


# Price by Carat and Clarity Comment 2: >You should see across facets the general movement of the points upwards and to the left, corresponding with smaller diamond sizes, but higher value for their sizes. As a final comment, did you remember to apply transformation functions to the price and carat values?</span>

# Let's try a different plot, for diamond price against cut and color quality features. To avoid the trap of higher quality grades being associated with smaller diamonds, and thus lower prices, we should focus our visualization on only a small range of diamond weights. For this plot, select diamonds in a small range around 1 carat weight. Try to make it so that your plot shows the effect of each of these categorical variables on the price of diamonds.

# In[9]:


# multivariate plot of price by cut and color, for approx. 1 carat diamonds

diamonds['carat'] = diamonds[(diamonds['carat'] < 1.05) & (diamonds['carat'] > 0.95)]
plt.figure(figsize=[10,7])
ax = sb.pointplot(data = diamonds, x = 'cut', y = 'price', hue = 'color', palette = 'viridis', dodge = 0.7, linestyles = "")


# Price by Cut and Color Comment 1: There's a lot of ways that you could plot one numeric variable against two categorical variables. I think that the clustered box plot or the clustered point plot are the best choices in this case. With the number of category combinations to be plotted (7x5 = 35), it's hard to make full sense of a violin plot's narrow areas; simplicity is better. A clustered bar chart could work, but considering that price should be on a log scale, there isn't really a nice baseline that would work well.</span>

# In[18]:


plt.figure(figsize = (16,7))
ax = sb.barplot(data = diamonds, x = 'color', y = 'price', hue = 'cut')


# Price by Cut and Color Comment 2: 
# Assuming you went with a clustered plot approach, you should see a gradual increase in price across the main x-value clusters, as well as generally upwards trends within each cluster for the third variable. Aesthetically, did you remember to choose a sequential color scheme for whichever variable you chose for your third variable, to override the default qualitative scheme? If you chose a point plot, did you set a dodge parameter to spread the clusters out? </span>

# In[ ]:


plt.figure(figsize = (16,7))
ax = sb.pointplot(data = diamonds, x = 'color', y = 'log_price', hue = 'cut')

