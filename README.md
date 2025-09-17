## How to use

```
!rm -rf data_science
!git clone https://github.com/GLI-Lab/data-science.git data_science
```

## Data

### Kaggle Datasets

https://www.kaggle.com/datasets?sort=votes

```python
# titanic
import seaborn as sns
titanic = sns.load_dataset("titanic")
titanic.to_csv("data_science/data/titanic.csv", index=False)

# California Housing Prices
https://www.kaggle.com/datasets/camnugent/california-housing-prices

# Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```
