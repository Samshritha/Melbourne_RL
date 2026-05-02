#!/bin/bash
set -e

pip install pandas numpy scikit-learn joblib --quiet

mkdir -p /data
mkdir -p /model
mkdir -p /environment

if [ ! -f "/environment/Melbourne_housing_FULL.csv" ]; then
    echo "ERROR: Melbourne_housing_FULL.csv not found in /environment/"
    exit 1
fi

cp /environment/Melbourne_housing_FULL.csv /data/Melbourne_housing_FULL.csv
python /environment/setup_data.py
cp /environment/broken_train.py /model/train.py
cp /environment/prompt.md /prompt.md

echo "Setup complete."
