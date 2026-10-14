# 原始数据目录

请用户自行从 Kaggle **Give Me Some Credit** 数据集页面下载并将以下文件放入本目录。当前实际文件名为：

- `GiveMeSomeCredit-training.csv`（代码也接受 `cs-training.csv`）
- `GiveMeSomeCredit-testing.csv`（代码也接受 `cs-test.csv`）
- `Data Dictionary.xls`

`GiveMeSomeCredit-testing.csv` 的目标列为空，不能用于项目模型评价。项目的训练、验证、测试三组数据从有标签的训练文件分层划分得到。

规则：

1. 项目不得自动下载这些文件。
2. 本目录中的原始文件视为只读，不得修改或覆盖。
3. 原始数据不得提交到 Git，也不得上传到第三方服务。
4. 后续处理结果应写入 `data/processed/` 或 `data/features/`。
