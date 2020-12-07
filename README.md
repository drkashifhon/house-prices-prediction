# House-prices-prediction
Predicting house prices using iphython jupyor notebook 

1 Fire up graphlab create
In [1]: import graphlab
2 Load some house sales data
In [2]: sales = graphlab.SFrame('home_data.gl/')
[INFO] This non-commercial license of GraphLab Create is assigned to chao@ou.eduand will expire on October [INFO] Start server at: ipc:///tmp/graphlab server-4127 - Server binary: /home/chao/anaconda/envs/dato-env/lib/python2.7/site-packages/graphlab/unity [INFO] GraphLab Server Version: 1.6.1
In [3]: sales
Out[3]: Columns:
id str
date datetime
price int
bedrooms str
bathrooms str
sqft living int
sqft lot int
floors str
waterfront int
view int
condition int
grade int
sqft above int
sqft basement int
yr built int
yr renovated int
zipcode str
lat float
long float
sqft living15 float
sqft lot15 float
Rows: 21613
Data:
+------------+---------------------------+---------+----------+-----------+-------------+
| id | date | price | bedrooms | bathrooms | sqft living |
+------------+---------------------------+---------+----------+-----------+-------------+
| 7129300520 | 2014-10-13 00:00:00+00:00 | 221900 | 3 | 1 | 1180 |
| 6414100192 | 2014-12-09 00:00:00+00:00 | 538000 | 3 | 2.25 | 2570 |
| 5631500400 | 2015-02-25 00:00:00+00:00 | 180000 | 2 | 1 | 770 |
| 2487200875 | 2014-12-09 00:00:00+00:00 | 604000 | 4 | 3 | 1960 |
| 1954400510 | 2015-02-18 00:00:00+00:00 | 510000 | 3 | 2 | 1680 |
| 7237550310 | 2014-05-12 00:00:00+00:00 | 1225000 | 4 | 4.5 | 5420 |
| 1321400060 | 2014-06-27 00:00:00+00:00 | 257500 | 3 | 2.25 | 1715 |
| 2008000270 | 2015-01-15 00:00:00+00:00 | 291850 | 3 | 1.5 | 1060 |
| 2414600126 | 2015-04-15 00:00:00+00:00 | 229500 | 3 | 1 | 1780 |
| 3793500160 | 2015-03-12 00:00:00+00:00 | 323000 | 3 | 2.5 | 1890 |
+------------+---------------------------+---------+----------+-----------+-------------+
+----------+--------+------------+------+-----------+-------+------------+---------------+
| sqft lot | floors | waterfront | view | condition | grade | sqft above | sqft basement |
+----------+--------+------------+------+-----------+-------+------------+---------------+
| 5650 | 1 | 0 | 0 | 3 | 7 | 1180 | 0 |
| 7242 | 2 | 0 | 0 | 3 | 7 | 2170 | 400 |
| 10000 | 1 | 0 | 0 | 3 | 6 | 770 | 0 |
| 5000 | 1 | 0 | 0 | 5 | 7 | 1050 | 910 |
| 8080 | 1 | 0 | 0 | 3 | 8 | 1680 | 0 |
| 101930 | 1 | 0 | 0 | 3 | 11 | 3890 | 1530 |
| 6819 | 2 | 0 | 0 | 3 | 7 | 1715 | 0 |
| 9711 | 1 | 0 | 0 | 3 | 7 | 1060 | 0 |
| 7470 | 1 | 0 | 0 | 3 | 7 | 1050 | 730 |
| 6560 | 2 | 0 | 0 | 3 | 7 | 1890 | 0 |
+----------+--------+------------+------+-----------+-------+------------+---------------+
+----------+--------------+---------+-------------+---------------+---------------+-----+
| yr built | yr renovated | zipcode | lat | long | sqft living15 | ... |
+----------+--------------+---------+-------------+---------------+---------------+-----+
| 1955 | 0 | 98178 | 47.51123398 | -122.25677536 | 1340.0 | ... |
| 1951 | 1991 | 98125 | 47.72102274 | -122.3188624 | 1690.0 | ... |
| 1933 | 0 | 98028 | 47.73792661 | -122.23319601 | 2720.0 | ... |
| 1965 | 0 | 98136 | 47.52082 | -122.39318505 | 1360.0 | ... |
| 1987 | 0 | 98074 | 47.61681228 | -122.04490059 | 1800.0 | ... |
| 2001 | 0 | 98053 | 47.65611835 | -122.00528655 | 4760.0 | ... |
| 1995 | 0 | 98003 | 47.30972002 | -122.32704857 | 2238.0 | ... |
| 1963 | 0 | 98198 | 47.40949984 | -122.31457273 | 1650.0 | ... |
| 1960 | 0 | 98146 | 47.51229381 | -122.33659507 | 1780.0 | ... |
| 2003 | 0 | 98038 | 47.36840673 | -122.0308176 | 2390.0 | ... |
+----------+--------------+---------+-------------+---------------+---------------+-----+
[21613 rows x 21 columns]
Note: Only the head of the SFrame is printed.
You can use print rows(num rows=m, num columns=n) to print more rows and columns.
3 Exploring the data for housing sales
In [4]: graphlab.canvas.set_target('ipynb')
sales.show(view = "Scatter Plot", x = "sqft_living",y = "price")
4 Create a simple regression model of sqft living to price
In [5]: train_data,test_data = sales.random_split(0.8,seed=0)
4.1 Build the regression model
In [6]: sqft_model = graphlab.linear_regression.create(train_data,target='price',features=['sqft_living'])
PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
You can set ``validation set=None`` to disable validation tracking.
PROGRESS: Linear regression:
PROGRESS: --------------------------------------------------------
PROGRESS: Number of examples : 16510
PROGRESS: Number of features : 1
PROGRESS: Number of unpacked features : 1
PROGRESS: Number of coefficients : 2
PROGRESS: Starting Newton Method
PROGRESS: --------------------------------------------------------
PROGRESS: +-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
PROGRESS: | Iteration | Passes | Elapsed Time | Training-max error | Validation-max error | Training-rmse PROGRESS: +-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
PROGRESS: | 1 | 2 | 1.005163 | 4348772.417409 | 2156925.341784 | 263383.770477 PROGRESS: +-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
# Evaluate the simple model
In [7]: print test_data['price'].mean()
543054.042563
In [8]: print sqft_model.evaluate(test_data)
f'max error': 4142954.698677999, 'rmse': 255192.00657238706g
5 Let's show what our predictions look like
In [9]: import matplotlib.pyplot as plt
%matplotlib inline
In [10]: plt.plot(test_data['sqft_living'],test_data['price'],'.',
test_data['sqft_living'],sqft_model.predict(test_data),'-')
Out[10]: [<matplotlib.lines.Line2D at 0x7f3004f4e610>,
<matplotlib.lines.Line2D at 0x7f3004f4e950>]
In [11]: sqft_model.get('coefficients')
Out[11]: Columns:
name str
index str
value float
Rows: 2
Data:
+-------------+-------+----------------+
| name | index | value |
+-------------+-------+----------------+
| (intercept) | None | -47219.8661536 |
| sqft living | None | 282.028833921 |
+-------------+-------+----------------+
[2 rows x 3 columns]
6 Explore other features in the data
In [12]: my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
In [13]: sales[my_features].show()
In [14]: sales.show(view='BoxWhisker Plot',x = 'zipcode',y = 'price')
7 Build a regression model with more features
In [15]: my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features)
PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
You can set ``validation set=None`` to disable validation tracking.
PROGRESS: Linear regression:
PROGRESS: --------------------------------------------------------
PROGRESS: Number of examples : 16497
PROGRESS: Number of features : 6
PROGRESS: Number of unpacked features : 6
PROGRESS: Number of coefficients : 115
PROGRESS: Starting Newton Method
PROGRESS: --------------------------------------------------------
PROGRESS: +-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
PROGRESS: | Iteration | Passes | Elapsed Time | Training-max error | Validation-max error | Training-rmse PROGRESS: +-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
PROGRESS: | 1 | 2 | 0.020630 | 3757139.114695 | 1017259.322510 | 183573.949009 PROGRESS: +-----------+----------+--------------+--------------------+----------------------+---------------+-----------------+
In [16]: print my_features
['bedrooms', 'bathrooms', 'sqft living', 'sqft lot', 'floors', 'zipcode']
In [17]: print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)
f'max error': 4142954.698677999, 'rmse': 255192.00657238706g
f'max error': 3518072.070271389, 'rmse': 179705.78017816716g
8 Apply learned models to predict prices of 3 houses
In [18]: house1 = sales[sales['id']=='5309101200']
In [19]: house1
Out[19]: Columns:
id str
date datetime
price int
bedrooms str
bathrooms str
sqft living int
sqft lot int
floors str
waterfront int
view int
condition int
grade int
sqft above int
sqft basement int
yr built int
yr renovated int
zipcode str
lat float
long float
sqft living15 float
sqft lot15 float
Rows: Unknown
Data:
+------------+---------------------------+--------+----------+-----------+-------------+
| id | date | price | bedrooms | bathrooms | sqft living |
+------------+---------------------------+--------+----------+-----------+-------------+
| 5309101200 | 2014-06-05 00:00:00+00:00 | 620000 | 4 | 2.25 | 2400 |
+------------+---------------------------+--------+----------+-----------+-------------+
+----------+--------+------------+------+-----------+-------+------------+---------------+
| sqft lot | floors | waterfront | view | condition | grade | sqft above | sqft basement |
+----------+--------+------------+------+-----------+-------+------------+---------------+
| 5350 | 1.5 | 0 | 0 | 4 | 7 | 1460 | 940 |
+----------+--------+------------+------+-----------+-------+------------+---------------+
+----------+--------------+---------+-------------+---------------+---------------+-----+
| yr built | yr renovated | zipcode | lat | long | sqft living15 | ... |
+----------+--------------+---------+-------------+---------------+---------------+-----+
| 1929 | 0 | 98117 | 47.67632376 | -122.37010126 | 1250.0 | ... |
+----------+--------------+---------+-------------+---------------+---------------+-----+
[? rows x 21 columns]
Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
You can use len(sf) to force materialization.
In [20]: print house1['price']
[620000, ... ]
In [21]: print sqft_model.predict(house1)
[629649.335256016]
In [22]: print my_features_model.predict(house1)
[720779.3868506234]
9 Prediction for a second, fancier house
In [23]: house2 = sales[sales['id']=='1925069082']
In [24]: house2
Out[24]: Columns:
id str
date datetime
price int
bedrooms str
bathrooms str
sqft living int
sqft lot int
floors str
waterfront int
view int
condition int
grade int
sqft above int
sqft basement int
yr built int
yr renovated int
zipcode str
lat float
long float
sqft living15 float
sqft lot15 float
Rows: Unknown
Data:
+------------+---------------------------+---------+----------+-----------+-------------+
| id | date | price | bedrooms | bathrooms | sqft living |
+------------+---------------------------+---------+----------+-----------+-------------+
| 1925069082 | 2015-05-11 00:00:00+00:00 | 2200000 | 5 | 4.25 | 4640 |
+------------+---------------------------+---------+----------+-----------+-------------+
+----------+--------+------------+------+-----------+-------+------------+---------------+
| sqft lot | floors | waterfront | view | condition | grade | sqft above | sqft basement |
+----------+--------+------------+------+-----------+-------+------------+---------------+
| 22703 | 2 | 1 | 4 | 5 | 8 | 2860 | 1780 |
+----------+--------+------------+------+-----------+-------+------------+---------------+
+----------+--------------+---------+-------------+---------------+---------------+-----+
| yr built | yr renovated | zipcode | lat | long | sqft living15 | ... |
+----------+--------------+---------+-------------+---------------+---------------+-----+
| 1952 | 0 | 98052 | 47.63925783 | -122.09722322 | 3140.0 | ... |
+----------+--------------+---------+-------------+---------------+---------------+-----+
[? rows x 21 columns]
Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
You can use len(sf) to force materialization.
In [25]: print sqft_model.predict(house2)
[1261393.9232383664]
In [26]: print my_features_model.predict(house2)
[1436834.3445759441]
In [63]: len(sales[(sales['sqft_living']>2000)&(sales['sqft_living']<=4000)])/(len(sales)*1.0)
Out[63]: 0.42187572294452413
In [35]: sales['sqft_living']
Out[35]: dtype: int
Rows: 21613
[1180, 2570, 770, 1960, 1680, 5420, 1715, 1060, 1780, 1890, 3560, 1160, 1430, 1370, 1810, 2950, In [38]: advanced_features =[
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors
'sqft_lot15', # average lot size of 15 nearest neighbors
]
10 Advanced feature model
In [39]: advanced_feature_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)
PROGRESS: Linear regression:
PROGRESS: --------------------------------------------------------
PROGRESS: Number of examples : 17384
PROGRESS: Number of features : 18
PROGRESS: Number of unpacked features : 18
PROGRESS: Number of coefficients : 127
PROGRESS: Starting Newton Method
PROGRESS: --------------------------------------------------------
PROGRESS: +-----------+----------+--------------+--------------------+---------------+
PROGRESS: | Iteration | Passes | Elapsed Time | Training-max error | Training-rmse |
PROGRESS: +-----------+----------+--------------+--------------------+---------------+
PROGRESS: | 1 | 2 | 0.027868 | 3469012.450663 | 154580.940735 |
PROGRESS: +-----------+----------+--------------+--------------------+---------------+
In [40]: print advanced_feature_model.evaluate(test_data)
f'max error': 3556849.413848093, 'rmse': 156831.11680191013g
In [41]: print my_features_model.evaluate(test_data)
f'max error': 3518072.070271389, 'rmse': 179705.78017816716g
In [58]: print sales[sales['zipcode']=='98039'].show()
None
In [66]: print my_features_model.evaluate(test_data)['rmse']-advanced_feature_model.evaluate(test_data)['rmse']
22874.6633763

