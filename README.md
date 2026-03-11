# 基于深度学习的光伏发电功率超短期预测
跨电站多源数据融合与特征工程；基于大数据分析的最优特征筛选；对 RNN、LSTM 和 GRU 三种循环网络的基础性能进行严格对比。

**1.设计流程图**

<img width="850" height="1353" alt="image" src="https://github.com/user-attachments/assets/8638bcef-c1f3-4285-88a1-e8bd4a19bc8a" />

**2.数据集介绍**

数据来源于https://www.kaggle.com/datasets/anikannal/solar-power-generation-data。其中包含两个真实的光伏电站的数据。

•	发电数据： 包括直流功率（DC Power）、交流功率（AC Power）以及日累计发电量（Daily Yield）。

•	气象数据： 包括实时辐照度（Irradiation）、环境温度（Ambient Temperature）以及模块温度（MODULE_TEMPERATURE）。

**3.数据加载与预处理**

eg:对两个光伏电站（Plant 1 和 Plant 2）的原始数据进行深度融合

通过基于时间戳的精确对齐，我们将两个独立电站的数据进行了合并，合并后的数据集总计 6416条 样本，时间跨度自 2020-05-15 至 2020-06-17。样本分布均衡，Plant 1 贡献了 3157 条数据（49.2%），Plant 2 贡献了 3259 条数据（50.8%），确保了模型在学习过程中不会产生电站偏倚。

•	完整性验证：经检测，合并后的数据集缺失值（Null Values）数量为 0。

•	数据清洁度：数据完整率达到 100.00%，无需进行复杂的插值补全，保证了原始物理信息的真实性与预测逻辑的可信度。

**4.基于时间序列的周期特征编码**

eg:光伏发电具有极强的日周期性。为了使深度学习模型能够有效识别时间维度的非线性规律，我们对小时特征（Hour）进行了周期性特征编码。

传统的线性小时表示（0-23）无法体现 23 点与次日 0 点的邻近性。为此，引入三角函数映射，即正弦编码、余弦编码，将时间信息映射至单位圆空间：

通过这种变换，模型可以更高效地捕捉光伏出力的日间循环模式，从而提升超短期预测在黎明和黄昏等功率突变时段的精度。

以下为两厂功率分布对比图

<img width="808" height="446" alt="image" src="https://github.com/user-attachments/assets/67269c58-2be2-45c1-941b-c1a660d1e732" />

**5.特征工程与大数据分析**

为使模型能有效学习，我们执行了以下关键的特征工程步骤：

• 时序特征构建: 为了捕捉时间的动态变化规律，我们构建了关键变量的历史时滞特征（如 IRRADIATION_lag_1），并对时间（小时）进行了周期性编码（如 Hour_sin, Hour_cos），帮助模型理解一天内的周期性模式。

• 多维度相关性分析: 我们采用Pearson（线性）、Spearman（单调）和Kendall（秩）三种方法，全面分析了各因素与未来不同时间尺度（5分钟至240分钟）功率输出的相关性。分析显示，辐照度（IRRADIATION）、模块温度（MODULE_TEMPERATURE）当前交流功率（AC_POWER）0.9041、0.8686和0.9578，表现出极强的短期预测价值。

• 特征重要性评估: 利用随机森林（Random Forest）和XGBoost两种强大的集成学习模型，我们对所有特征的重要性进行了量化评估。结果一致表明，“IRRADIATION”（辐照度）是决定功率输出的决定性输入特征，其在随机森林模型中的重要性评分为0.719305，在XGBoost中为0.332498，均排名第一。

• 多时间尺度影响分析: 通过分析各特征与不同预测时长目标的相关性，我们评估了其影响力的衰减规律。综合来看，AC_POWER、Hour_cos（小时的余弦编码）和IRRADIATION在多个时间尺度上均表现出最强且最稳定的平均影响力。
综合上述相关性分析和重要性评估的结果，我们筛选出对模型预测最有价值的特征子集。

我们确定了以下9个特征构成的最优特征集合，用于后续的模型训练： ['AC_POWER', 'AMBIENT_TEMPERATURE', 'DAILY_YIELD', 'DC_POWER', 'Hour', 'Hour_cos', 'Hour_sin', 'IRRADIATION', 'MODULE_TEMPERATURE']

表5-1 特征重要性与相关性综合评估表

特征名称	平均相关性	RF 重要性排名	XGB 重要性排名

AC_POWER	0.6386	2	2

Hour_cos	0.6355	9	9

IRRADIATION	0.6187	6	6

MODULE_TEMPERATURE	0.5969	5	20

DC_POWER	0.4998	1	27

AMBIENT_TEMPERATURE	0.3817	4	16

Hour_sin	0.3641	41	38

DAILY_YIELD	0.2697	33	30

Hour	0.1874	41	38

**6.多输出深度学习模型设计与对比**

选择正确的模型架构是实现高精度预测的核心环节。在选择出合理的特征之后，我们聚焦于处理时间序列数据的循环神经网络（RNN）及其高级变体，如何通过严谨的实验对比，从RNN、长短期记忆网络（LSTM）和门控循环单元（GRU）三种主流架构中，选出最适合本预测任务的基础模型。
性能对比指标：RMSE、MAE、MAPE、R²

通过实证对比确定了RNN为最优基础模型。

表3-3 RNN最终性能评估结果

预测时长	RMSE (W)	MAE (W)	R² Score

5分钟	2170.84	1254.28	0.8599

15分钟	2573.99	1459.32	0.8022

30分钟	2573.99	1488.77	0.8018

**7.预测结果对比**

<img width="866" height="605" alt="image" src="https://github.com/user-attachments/assets/a21a4cc7-aaf6-4634-9d5d-26ac9ed9ebc1" />

<img width="865" height="576" alt="image" src="https://github.com/user-attachments/assets/913444f5-c902-4405-b4f5-5296e958300e" />

**8.与基准线模型的对比**

我们选择了三种具有代表性的模型作为基准，覆盖了从简单统计到经典机器学习的不同技术路线：历史平均法 (Historical Mean)、随机森林 (Random Forest)、浅层MLP (Shallow MLP)

<img width="865" height="605" alt="image" src="https://github.com/user-attachments/assets/3c475c65-4049-41c0-8000-eae45c9768bb" />

5分钟预测 - RNN 相对于基准线的改进:

• 相对历史平均法: 64.1%

• 相对随机森林:   -4.7%

• 相对浅层MLP:   0.8%

15分钟预测 - RNN 相对于基准线的改进:

• 相对历史平均法: 57.3%

• 相对随机森林:   -0.0%

• 相对浅层MLP:   -1.9%

30分钟预测 - RNN 相对于基准线的改进:

• 相对历史平均法: 57.3%

• 相对随机森林:   5.0%

• 相对浅层MLP:   -4.0%

**9.相关参数设置**

配置项目	参数值

模型类型 	RNN 

输入特征数 	10

隐藏层维度 	128

循环层层数 	3

总参数量 	94,788

优化器	Adam

损失函数	MSE

类别	样本数量	占比	批次数 

训练集 (Train Set)	4,447	70.0%	139

验证集 (Val Set)	953	15.0%	30

测试集 (Test Set)	954	15.0%	30

总计 (Sum)	6,354	100%	199


