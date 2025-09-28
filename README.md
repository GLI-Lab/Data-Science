## How to use

```
!rm -rf data_science
!git clone https://github.com/GLI-Lab/data-science.git data_science
```

## Data

### Overview

| Dataset | Type | Primary Task | Target Variable |
|---|---|---|---|
| `Amtrak.csv` | Time series | Regression | `Ridership` |
| `BostonHousing.csv` | Tabular | Regression | `MEDV` |
| `california_housing.csv` | Tabular | Regression | `median_house_value` |
| `titanic.csv` | Tabular | Classification | `Survived` |
| `california_cities.csv` | Tabular | Regression | `population` |

### Source & Loader

https://www.kaggle.com/datasets?sort=votes

```python
# titanic
import seaborn as sns
titanic = sns.load_dataset("titanic")
titanic.to_csv("./data/titanic.csv", index=False)

# California Housing Prices
https://www.kaggle.com/datasets/camnugent/california-housing-prices

# Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# MUTAG
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
dataset = TUDataset(
    root='./data', name='MUTAG', use_node_attr=False, cleaned=True, transform=T.Compose([
        T.LocalDegreeProfile(),
    ])
)
```
