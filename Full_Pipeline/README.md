This directory contains the complete end-to-end implementation of the modelling pipeline, including model definitions, training and evaluation entrypoints, configuration files and testing. This is intended to be run as a standalone pipeline using predefined configurations.

Data
Datasets need to be provided as .csv files and placed in the data/ directory.
Synthetic datasets can be generated using the notebooks provided in ShuhanZhang_repo/.
Expedia datasets can be obtained from Kaggle. No feature engineering or transformation was performed on the Expedia dataset beyond splitting the base dataset into training and test sets.

Checkpoints
Trained model checkpoints are written to the checkpoints/ directory.
Both data/ and checkpoints/ are excluded from version control.

Execution
All runs are driven by YAML configuration files.

Training:
python train.py --config configs/<train_config>.yaml

Continue training:
python continue_train.py --config configs/<continue_train_config>.yaml

Evaluation:
python evaluate.py --config configs/<evaluate_config>.yaml

The configs/ directory contains examples of the standard configuration format and should be consulted when creating new runs.

Testing
Pipeline-level pytest tests are provided under test/ and can be run independently to validate training and evaluation logic.
