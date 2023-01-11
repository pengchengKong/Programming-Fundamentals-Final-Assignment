# Comparison of ARIMA model and LSTM model for predicting the number of confirmed covid-19 cases in Hong Kong

Novel coronavirus pneumonia is an emerging acute respiratory infectious disease. Since December 2019, the new coronavirus pneumonia epidemic has continued to spread globally, endangering the lives and health of people around the world, putting the public health care system to a severe test, and at the same time, causing a huge impact on economic and trade activities and seriously hindering the development of society.

Since the announcement of the relaxation of the epidemic prevention and control policy in Hong Kong on April 1, the number of new confirmed cases has continued to increase significantly, and the curve of the cumulative number of confirmed cases over time shows a certain pattern. By using statistical methods to model the spread of the new coronavirus and the development of the epidemic, we can provide important references for the formulation of response measures. In particular, at this stage, the epidemic prevention and control policy has been adjusted to "category B, category B control", and full nucleic acid testing is no longer conducted.



## Operating Environment 

``` r
Python 3.9.7 [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
```



## Reproduction

## **1.ARIMA模型建模**

**①预处理**：

i.***\*准确性审核\****：检查时间序列数据是否存在异常值，利用四分位差IQR（=QU-QL），把超过IQR1.5倍距离的数值定义为离群点，把超过IQR3倍距离的数值定义为极端值。通过panda中的mask函数将离群点替换为缺失值。

```
IQR = data['confirmedCount'].quantile(0.75) - data['confirmedCount'].quantile(0.25)
IQRL = data['confirmedCount'].quantile(0.25) - 3*IQR
IQRU = data['confirmedCount'].quantile(0.75) + 3*IQR
data["confirmedCount"] = data["confirmedCount"].mask((data["confirmedCount"] < IQRL) | (data["confirmedCount"] > IQRU), np.nan)
```

ii.***\*完整性审核\****：检查时间序列数据是否存在缺失值，通过k-Nearest Neighbors算法对缺失值进行插补。

```
print(data.isna().sum())
imputer = KNNImputer(n_neighbors=4) #邻居样本求平均数
data=imputer.fit_transform(data)
```

ii.对时间序列进行平稳性、纯随机性检验，若为平稳非白噪声序列，则采用ARMA模型进行建模。

```
data.plot(figsize=(15, 8), fontsize=12)
plt.xlabel('date', fontsize=12)
plt.ylabel('confirmedCount', fontsize=12)
plt.title("香港2022年COVID-19确诊人数")
plt.show()
```

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps8.jpg) 

根据时间序列线图进行平稳性检验，确诊数据随时间变化存在明显的非线性趋势，对序列进行1阶差分运算提取确定性信息。

```
data["confirmedCount_df"] = data["confirmedCount"].diff(1)
data["confirmedCount_df"].plot(figsize=(15, 8), fontsize=12)
plt.xlabel('date', fontsize=12)
plt.ylabel('diffconfirmedCount', fontsize=12)
plt.title("香港2022年COVID-19确诊人数差分数据-每日新增确诊")
plt.show()
```

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps9.jpg) 

**②模型识别**：

i.***\*平稳性检验\****：对差分后的序列进行ADF平稳性检验，通过检验自回归方程的自回归系数之和是否等于1，判断是否存在特征根，考察序列平稳性，若拒绝原假设则表明序列平稳：

```
print(adfuller(data["confirmedCount_df"].dropna()))

(-2.8124662101765314, 0.0565150048748737, 17, 343, {'1%': -3.449559661646851, '5%': -2.8700035112469626, '10%': -2.5712790073013796}, 6660.549174635945)
```

 

ii.***\*纯随机性检验\****：对差分后的序列进行LB纯随机性检验，判断序列值之间是否存在相关关系，以判断历史数据对未来发展有无影响，若无相关关系则序列无分析价值，若拒绝原假设则表明序列为非白噪声序列：

```
print(acorr_ljungbox(data["confirmedCount_df"].dropna()))

​    lb_stat    lb_pvalue
1   312.926115  5.032685e-70
2   598.835869  9.213894e-131
3   852.217124  2.047042e-184
4  1076.558668  9.121562e-232
5  1267.386173  7.426754e-272
6  1423.336388  2.143759e-304
```



平稳性检验和纯随机性检验显示该序列为平稳非白噪声序列，可以使用ARIMA模型拟合该序列，根据该时间序列的样本自相关系数(ACF)和样本偏自相关系数(PACF)特征，选择阶数适当的ARIMA(p,d,q)模型进行

```
acf = plot_acf(data["confirmedCount_df"].dropna(), lags=40)
plt.title('差分后数据的自相关图')
acf.show()
```

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps10.jpg) 

```
acf = plot_acf(data["confirmedCount_df"].dropna(), lags=40)
plt.title('差分后数据的自相关图')
acf.show()
```

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps11.jpg) 

依据Bartlett定理：![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps12.jpg)和Quenouille定理：![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps13.jpg)，由正态分布性质，若ACF或PACF在最初d阶明显超过2倍标准差范围，而后几乎95%都落在2倍标准差范围内，且由非0系数衰减为小值波动的过程非常突然，则认为ACF或PACF d阶截尾，若有超过5%的ACF或PACF落在2倍标准差范围外，或由非0系数衰减为小值波动的过程比较缓慢或非常连续，则认为ACF或PACF拖尾，同时根据BIC准则，选择使BIC最小的自相关阶数和偏自相关阶数，将模型定阶为ARIMA（4,1,4)。

```
p_min = 0
d_min = 0
q_min = 0
p_max = 5
d_max = 1
q_max = 5
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
for p,d,q in itertools.product(range(p_min,p_max+1),range(d_min,d_max+1),range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    try:
        model = sm.tsa.ARIMA(data["confirmedCount_df"].dropna(), order=(p, d, q))
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,mask=results_bic.isnull(),ax=ax,annot=True,fmt='.2f')
ax.set_title('BIC')
plt.show()
```

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps14.jpg) 

 

**③模型参数估计**：

i.***\*矩估计\****：计算量小、估计思想简单直观且不需要假设总体分布，但只用到p+q个样本自相关系数，忽略了序列中其他信息，估计精度不高。

ii.***\*条件最小二乘估计\****：假定过去未观测到的序列值为0，则未知参数是使得残差平方和达到最小的估计值，也充分利用了序列中样本信息，估计精度很高。

iii.***\*极大似然估计\****：假定序列服从多元正态分布，在极大似然准则下认为未知参数是使得似然函数达到最大的参数值，充分利用了序列中样本信息，估计精度高，且估计量具有一致性、渐近正态性、渐近有效性。

```
model=sm.tsa.ARIMA(data["confirmedCount_df"].dropna(),order=(4,1,4)).fit()
model.summary()
```

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps15.jpg) 

**④****模型检验**：

```
#模型有效性检验
resid = model.resid
#对残差进行正态性检验
qqplot(resid, line='q', fit=True)
plt.show()
#对残差进行白噪声检验
print(acorr_ljungbox(resid))
```

i.***\*模型的显著性/有效性检验\****

通过检验残差序列是否为白噪声序列，判断其提取序列中样本信息是否充分，模型是否显著有效拟合观察值序列的波动。

a.建立假设：![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps16.jpg)，![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps17.jpg)

b.检验统计量：![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps18.jpg)

c.统计决策：当LB统计量大于临界值或P值小于α时，拒绝原假设，认为残差序列为非白噪声序列，拟合模型不显著，重新选择模型拟合；当LB统计量小于临界值或P值大于α时，不能拒绝原假设，认为残差序列为白噪声序列，拟合模型显著有效。

ii.***\*残差的正态性检验\****：根据QQ图上的点是否都大致落在标准正态分布线上，判断数据是否服从正态分布。

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps19.jpg) 

iii.***\*参数的显著性检验\****

通过检验每一个未知参数是否显著非0，剔除模型中不显著非0参数对应的自变量，使精简为由对因变量影响明显的自变量表示的模型。

a.建立假设：![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps20.jpg)

b.检验统计量：![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps21.jpg)

c.当|T|大于临界值或P值小于α时，拒绝原假设，认为参数显著非0；当|T|小于临界值或P值大于α时，不拒绝原假设，认为参数不显著非0，剔除其对应的自变量重新拟合模型。

 

**⑤****模型预测**：

利用训练好,通过检验的最优模型对时间序列的未来发展作出预测，通过RMSE损失函数估计模型的预测值与真实值的不一致程度，损失函数越小，模型的鲁棒性就越好，模型拟合效果越好，模型均方误差RMSE: 3481.623712。

```
#模型预测
predict_data = model.predict(1,360)
print('拟合数据')
print(predict_data)

forecast_data = model.forecast(6)
print('预测数据')
print(forecast_data)

fig = plt.subplots(1,1,figsize=(10,7),dpi=100)
plt.plot(predict_data, label='拟合数据')
plt.plot(forecast_data, label='预测数据')
plt.legend()
plt.title("ARIMA模型下香港2022年COVID-19每日新增人数拟合与预测图")
plt.show()
```

​       Point Forecast    真实值

 2022/12/29   2568971     2568596

 2022/12/30   2587593     2596426

 2022/12/31   2605907     2625633

 2022/01/01   2624301     2648994

 2022/01/02   2642601     2669224

 

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps22.jpg) 

 

![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps23.jpg) 

## **2.LSTM模型建模**



![img](file:///C:\Users\16598\AppData\Local\Temp\ksohtml14948\wps24.jpg) 
