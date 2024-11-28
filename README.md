# FCL4DD

This is the implement of FCCL4MDD. We trained with Derm7pt and JSIEC dataset.

## Install Requirement

opencv>==6.5.5
torch>=1.7.0
python>=3.6
scikit-fuzzy>=0.4.2
scikit-learn>=0.24.2

## How To Run


#### Step 1: Download dataset

Download the Derm7pt and JSIEC datasets to the data\ directory and split them into Train and Test sets with a 7:3 ratio.

#### Step 2: Training

python3 main.py --seed 2025 --dataset dataset/JSIEC --train_data path/to/train --test_data path/to/test --com_round 100 --tasks5 --beta 0
