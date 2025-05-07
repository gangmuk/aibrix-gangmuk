#!/bin/bash

python train_and_test.py processed_dataset.csv

python inference.py latency_predictor.joblib test_dataset.csv predictions.csv