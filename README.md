

### crop_43数据集

将6个类别中各取10%的样本按照比例p与'Painted metal sheets'类别光谱融合，构成新的第7个类别'Landmine'，使用a%的样本训练并用b%的样本测试
- 表格1

|         实验配置          | Landmine分类准确率 | precision | recall |       tp, fp, tn, fn       |
| :-----------------------: | :----------------: | :-------: | :----: | :------------------------: |
|  p = 1; a = 0.9; b = 0.1  |       85.83        |   98.04   | 98.59  |    3556, 71, 71322, 51     |
|    p = 1（全新数据集）    |                    |   91.15   | 89.41  |  3225， 313， 71080， 382  |
|                           |                    |           |        |                            |
| p = 0.8; a = 0.9; b = 0.1 |       84.17        |   98.69   | 98.42  |   3550， 47， 71346， 57   |
|   p = 0.8（全新数据集）   |                    |   90.12   | 88.00  |  3174， 348， 71045， 433  |
|                           |                    |           |        |                            |
| p = 0.5; a = 0.9; b = 0.1 |       82.78        |   96.28   | 98.28  |    3435, 137, 71256, 62    |
|   p = 0.5（全新数据集）   |                    |   85.60   | 85.72  |  3092， 520， 70873， 515  |
|                           |                    |           |        |                            |
| p = 0.2; a = 0.9; b = 0.1 |       34.72        |   74.27   | 70.81  | 2554， 885， 70508， 1053  |
|   p = 0.2（全新数据集）   |                    |   54.19   | 42.89  | 1547， 1308， 70085， 2060 |
