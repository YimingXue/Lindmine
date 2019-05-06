### aviaU

- ResNetv2网络处理PaviaU数据集，75%测试（每一类最多选取200个样本训练）、25%测试，lr=0.01，每一个batch_size（100次迭代）记录一次loss，网络使用ResNetv2，都添加了Focal Loss1

  - P: Patch_size
  - NCB: without CBLoss
  - NF: without FC
  - ResNetv3_CBLoss
  - ND: without Dropout
  - CE: contrast experiment
  - CB Loss

  |     Class      |  P15   |  P21  |      | P27NCB | P27NF  | ResNetv3_CBLoss | P27ND  | P23CE  |
  | :------------: | :----: | :---: | :--: | :----: | :----: | :-------------: | :----: | :----: |
  |       1        | 91.55  | 97.71 |      | 98.79  | 97.77  |      98.43      | 97.95  | 99.52  |
  |       2        | 96.10  | 98.54 |      | 99.38  | 97.15  |      98.63      | 98.97  | 99.85  |
  |       3        | 96.18  | 93.13 |      | 99.24  | 98.28  |      99.62      | 98.85  | 99.62  |
  |       4        | 98.04  | 97.65 |      | 98.56  | 95.56  |      97.65      | 97.52  | 99.61  |
  |       5        | 100.00 | 99.70 |      | 99.40  | 100.00 |     100.00      | 100.00 | 99.40  |
  |       6        | 99.28  | 99.36 |      | 98.97  | 98.17  |      99.36      | 99.05  | 100.00 |
  |       7        | 97.89  | 99.70 |      | 100.0  | 100.00 |     100.00      | 100.00 | 100.00 |
  |       8        | 96.63  | 98.26 |      | 98.59  | 98.48  |      98.80      | 98.48  | 99.78  |
  |       9        | 99.58  | 99.58 |      | 100.0  | 99.58  |      99.15      | 98.31  | 99.15  |
  |                |        |       |      |        |        |                 |        |        |
  |       OA       | 96.21  | 98.25 |      | 99.14  | 97.65  |      98.77      | 98.72  | 99.76  |
  |       AA       | 97.25  | 98.18 |      | 99.21  | 98.33  |      99.07      | 98.79  | 99.66  |
  |     Kappa      | 95.04  | 97.69 |      | 98.86  | 96.92  |      98.38      | 98.31  | 99.68  |
  | 运行时间/epoch | 1.27s  | 1.55s |      |        |        |                 |        |        |

  - CB: Category Balanced Loss

  |     Class      | CB0.5  | P27CB1 | CB1.5  | CB2.0  | CB2.5  |
  | :------------: | :----: | :----: | :----: | :----: | :----: |
  |       1        | 98.73  | 98.85  | 98.67  | 98.25  | 97.16  |
  |       2        | 99.40  | 99.40  | 99.27  | 98.78  | 98.16  |
  |       3        | 100.00 | 99.24  | 99.24  | 99.05  | 96.18  |
  |       4        | 97.78  | 98.30  | 98.56  | 97.39  | 97.00  |
  |       5        | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
  |       6        | 99.36  | 99.44  | 99.44  | 99.05  | 99.36  |
  |       7        | 100.00 | 100.00 | 99.70  | 100.00 | 99.70  |
  |       8        | 99.02  | 98.91  | 98.48  | 98.70  | 98.80  |
  |       9        | 99.58  | 100.00 | 99.15  | 99.58  | 97.88  |
  |                |        |        |        |        |        |
  |       OA       | 99.21  | 99.24  | 99.11  | 98.73  | 98.12  |
  |       AA       | 99.32  | 99.35  | 99.17  | 98.98  | 98.25  |
  |     Kappa      | 98.96  | 99.00  | 98.83  | 98.32  | 97.52  |
  | 运行时间/epoch |        |        |        |        |        |



### Indian_pines

- ResNetv2网络处理Indian_pines数据集，75%测试（每一类最多选取200个样本训练）、25%测试，lr=0.01，每一个batch_size（100次迭代）记录一次loss，网络使用ResNetv2

  - P: Patch_size=29
  - NFL: without Focal Loss
  - CE: contrast experiment
  - CB Loss

  | Class |   CE   | P29CB1 |
  | :---: | :----: | :----: |
  |   1   | 100.00 | 100.00 |
  |   2   | 98.04  | 99.44  |
  |   3   | 100.00 | 99.52  |
  |   4   | 100.00 | 100.00 |
  |   5   | 99.17  | 98.33  |
  |   6   | 100.00 | 100.00 |
  |   7   | 100.00 | 100.00 |
  |   8   | 100.00 | 100.00 |
  |   9   | 100.00 | 100.00 |
  |  10   | 98.35  | 99.59  |
  |  11   | 96.08  | 98.21  |
  |  12   | 100.00 | 100.00 |
  |  13   | 100.00 | 100.00 |
  |  14   | 99.68  | 99.68  |
  |  15   | 100.00 | 100.00 |
  |  16   | 100.00 | 100.00 |
  |       |        |        |
  |  OA   | 98.54  | 99.30  |
  |  AA   | 99.46  | 99.67  |
  | Kappa | 98.34  | 99.20  |