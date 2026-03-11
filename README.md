# 基于深度学习的光伏发电功率超短期预测
跨电站多源数据融合与特征工程；基于大数据分析的最优特征筛选；对 RNN、LSTM 和 GRU 三种循环网络的基础性能进行严格对比。
**1.设计流程图**
<img width="850" height="1353" alt="image" src="https://github.com/user-attachments/assets/8638bcef-c1f3-4285-88a1-e8bd4a19bc8a" />

**2.数据集介绍**
数据来源于https://www.kaggle.com/datasets/anikannal/solar-power-generation-data。其中包含两个真实的光伏电站的数据。
•	发电数据： 包括直流功率（DC Power）、交流功率（AC Power）以及日累计发电量（Daily Yield）。
•	气象数据： 包括实时辐照度（Irradiation）、环境温度（Ambient Temperature）以及模块温度（MODULE_TEMPERATURE）。
**3.数据加载与预处理**
eg:对两个光伏电站（Plant 1 和 Plant 2）的原始数据进行深度融合与特征工程
通过基于时间戳的精确对齐，我们将两个独立电站的数据进行了合并，合并后的数据集总计 6416条 样本，时间跨度自 2020-05-15 至 2020-06-17。样本分布均衡，Plant 1 贡献了 3157 条数据（49.2%），Plant 2 贡献了 3259 条数据（50.8%），确保了模型在学习过程中不会产生电站偏倚。
•	完整性验证：经检测，合并后的数据集缺失值（Null Values）数量为 0。
•	数据清洁度：数据完整率达到 100.00%，无需进行复杂的插值补全，保证了原始物理信息的真实性与预测逻辑的可信度。
**4.基于时间序列的周期特征编码**
光伏发电具有极强的日周期性。为了使深度学习模型能够有效识别时间维度的非线性规律，我们对小时特征（Hour）进行了周期性特征编码
传统的线性小时表示（0-23）无法体现 23 点与次日 0 点的邻近性。为此，本研究引入三角函数映射，将时间信息映射至单位圆空间：
正弦编码： {\mathrm{Hour}}_{\mathrm{sin}}=\mathbf{si}\mathbf{n}{\left(\frac{\mathbf{2\pi}\cdot\mathrm{Hour}}{\mathbf{24}}\right)}
余弦编码： {\mathrm{Hour}}_{\mathrm{cos}}=\mathbf{co}\mathbf{s}{\left(\frac{\mathbf{2\pi}\cdot\mathrm{Hour}}{\mathbf{24}}\right)}
通过这种变换，模型可以更高效地捕捉光伏出力的日间循环模式，从而提升超短期预测在黎明和黄昏等功率突变时段的精度。
以下为两厂功率分布对比图
<img width="808" height="446" alt="image" src="https://github.com/user-attachments/assets/67269c58-2be2-45c1-941b-c1a660d1e732" />
**5.特征工程与大数据分析**
