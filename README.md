## 指标论证实验方案

#### 实验目的和任务

##### 实验目的

本次实验的目的是为了论证：抛撒式地雷尺寸（12 cm×5 cm左右）与高光谱仪空间分辨率（12.5 cm×12.5 cm或更高）比例在什么范围内时，即空间分辨率在什么范围内时，分类算法能将地雷与背景区分出来。

#####  实验任务

​    本次实验任务如下：

1. 截取Shandong Downtown数据集中一部分图像进行实验；
2. 选定一个不存在于该数据集中的类别，用来代替地雷，将其谱带信息以一定比例逐个像素地融入数据集中其他类别部分样本的谱带中，分别研究不同融入比例对分类效果的影响。



#### 实验流程

1. 在尺寸为2000×2700的Shandong Downtown数据集上裁剪出尺寸为250×300部分（下文提到的数据集都是指代裁剪出的这部分），裁剪范围为高度750:1000，宽度1000:1300。数据集中一共包括了6种地物类型。

   | 编号 |          类别          |
   | :--: | :--------------------: |
   |  1   |         Trees          |
   |  3   |        Shadows         |
   |  6   |          Cars          |
   |  9   |    Roofs type Tree     |
   |  13  | Cement floors type Two |
   |  14  |          Soil          |

2. 将不存在于数据集中的“Painted metal sheets”类别代替抛撒式地雷，将“Painted metal sheets”按照比例p融入其他6个类别各选出的10%样本中，并生成的新的类别“landmine”，因此数据集中总共有7个类别，并修正它们的类别为1~7；

   | 编号 |          类别          |
   | :--: | :--------------------: |
   |  1   |         Trees          |
   |  2   |        Shadows         |
   |  3   |          Cars          |
   |  4   |    Roofs type Tree     |
   |  5   | Cement floors type Two |
   |  6   |          Soil          |
   |  7   |        Lindmine        |

3. 对数据集中的7个类别分别选取90%的样本训练和10%样本测试，测试不同融入比例p对分类结果的影响；

4. 不同比例下第一行实验结果测试的是包含训练数据部分的结果，第二行实验结果为相同配置下全新数据集的分类结果。



#### 实验初步结果

|     融入比例p      | Landmine分类准确率 | precision | recall |       tp, fp, tn, fn       |
| :----------------: | :----------------: | :-------: | :----: | :------------------------: |
|        100%        |       85.83        |   98.04   | 98.59  |    3556, 71, 71322, 51     |
| 100%（全新数据集） |                    |   91.15   | 89.41  |  3225， 313， 71080， 382  |
|                    |                    |           |        |                            |
|        80%         |       84.17        |   98.69   | 98.42  |   3550， 47， 71346， 57   |
| 80%（全新数据集）  |                    |   90.12   | 88.00  |  3174， 348， 71045， 433  |
|                    |                    |           |        |                            |
|        50%         |       82.78        |   96.28   | 98.28  |    3435, 137, 71256, 62    |
| 50%（全新数据集）  |                    |   85.60   | 85.72  |  3092， 520， 70873， 515  |
|                    |                    |           |        |                            |
|        20%         |       34.72        |   74.27   | 70.81  | 2554， 885， 70508， 1053  |
| 20%（全新数据集）  |                    |   54.19   | 42.89  | 1547， 1308， 70085， 2060 |



#### 实验结果说明

由实验可以看到随着“Painted metal sheets”融入比例逐步下降，分类的准确率和召回率也相应下降。如果将比例p看作是抛撒式地雷与空间分辨率的比值，那么为了保证90以上的分类准确率和近90的召回率，需要地雷尺寸占80%的空间分辨率长度。

抛撒式地雷大小大致为12 cm×5 cm，选取短边5 cm，空间分辨率尽量要在6.25 cm左右。而且这只是仿真结果，数据相对干净，实际应用中，空间分辨率希望可以到达5 cm。

因此我们得出的结论是空间分辨率尽量是要在5-7 cm，这个范围内都对抛撒式地雷都有一定的检测能力。