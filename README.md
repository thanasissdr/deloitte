# INSTRUCTIONS


## Setting up a virtual environment

Create a (conda) virtual environment, e.g.
```cmd
conda create -n project python=3.10.8 ipykernel
conda activate project
```

### Installing necessary packages

> If tensorflow utilises GPU:
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

While in the `$ROOT_DIR`, i.e. `project`:
```cmd
pip install -r requirements.txt
pip install -e .
```

## Write csv files into feather format (faster to load)

```cmd
python helpers/csv_to_feather.py
```
## Exploratory Data Analysis (EDA)

Please run the notebook inside the `eda` folder, which includes some basic analysis.

## Modelling
Please run the notebooks inside the `modelling` folder.