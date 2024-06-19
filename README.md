# NLCC 2024 Shared Task 2: Nominal Compound Chain Extraction

[切换到中文](README_zh.md)

Welcome to the Nominal Compound Chain Extraction project for the NLCC 2024 Shared Task 2 competition. This project contains the necessary files and instructions to train, validate, and test a model for extracting nominal compound chains from given datasets.

## Project Structure

```
.
├── data
│   └── nlpcc_data
│       ├── train.json
│       ├── valid.json
│       └── test.json
│   └── submit
│   └── save
├── src
├── inference.py
├── main.py
├── README.md
└── requirements.txt
```

- `data/`: Directory containing the dataset.
  - `nlpcc_data/`: Subdirectory containing the train, validation, and test JSON files.
  - `save/`: Subdirectory containing the model weights.
  - `submit/`: Subdirectory containing the predicted JSON files.
- `src`: Directory containing the source code.
- `inference.py`: Script to load the best performing model and obtain prediction results on the test set.
- `main.py`: Script to train the model and obtain the best model on the validation set.
- `README.md`: This file.
- `requirements.txt`: File containing the required Python packages.

## Dataset

- `train.json`: Contains the training data with corpus and labels.
- `valid.json`: Contains the validation data with corpus and labels.
- `test.json`: Contains the test data with document ID and corpus, but without labels.

## Setup

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using the following command:

    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

To train the model and obtain the best model on the validation set, run the following command:

```sh
python main.py
```

The best model will be saved in the `data/save/` directory.

## Inference

To load the best performing model and obtain the prediction results on the test set, run the following command:

```sh
python inference.py --chk best_12.pth.tar
```
Please replace `best_12.pth.tar` with the name of the best model file.

The prediction results will be saved in the `data/submit/submit.jsonl` file.

## Submission

After obtaining the prediction results, you can zip the `submit.jsonl` file and
upload the zippped file to the [competition platform](https://www.codabench.org/competitions/3179/) to obtain your score.

## License

This project is licensed under the Apache License 2.0 License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any issues or questions, please contact us.

Good luck with the competition!
