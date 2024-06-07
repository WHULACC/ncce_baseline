# 中文

[Switch to English](README.md)

欢迎参加 NLCC 2024 共享任务 2 名词复合链提取项目。该项目包含训练、验证和测试模型以从给定数据集中提取名词复合链所需的文件和说明。

## 项目结构

```
.
├── data
│   └── nlpcc_data
│       ├── train.json
│       ├── valid.json
│       └── test.json
│   └── submit
│   └── save
├── inference.py
├── main.py
├── README.md
└── requirements.txt
```

- `data/`: 包含数据集的目录。
  - `nlpcc_data/`: 包含训练、验证和测试 JSON 文件的子目录。
  - `save/`: 包含模型权重的子目录。
  - `submit/`: 包含预测 JSON 文件的子目录。
- `src`: 包含模型代码的目录。
- `inference.py`: 加载最佳模型并对测试集进行预测的脚本。
- `main.py`: 训练模型并在验证集上获得最佳模型的脚本。
- `README.md`: 本文件。
- `requirements.txt`: 包含所需 Python 包的文件。

## 数据集

- `train.json`: 包含带有语料和标签的训练数据。
- `valid.json`: 包含带有语料和标签的验证数据。
- `test.json`: 包含文档 ID 和语料但没有标签的测试数据。

## 设置

1. 将仓库克隆到本地计算机。
2. 导航到项目目录。
3. 使用以下命令安装所需的包：

    ```sh
    pip install -r requirements.txt
    ```

## 训练模型

要训练模型并在验证集上获得最佳模型，请运行以下命令：

```sh
python main.py
```

最佳模型将保存在 `data/save/` 目录下。

## 推理

要加载最佳模型并对测试集进行预测，请运行以下命令：

```sh
python inference.py --chk best_12.pth.tar
```
请将`best_12.pth.tar` 替换为您的最佳模型文件名。

预测结果将保存在 `data/submit/submit.jsonl` 文件中。

## 提交

获得预测结果后，将`submit.jsonl`压缩为zip格式后，上传到[比赛平台](https://www.codabench.org/competitions/3179)以获得分数。
每支队伍在比赛期间最多有5次提交机会。

## 许可

本项目根据 Apache License 2.0 许可发布。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题或需要帮助，可以联系我们。

祝你在比赛中取得好成绩！