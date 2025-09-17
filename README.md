## How to use

```
!rm -rf data_science
!git clone https://github.com/GLI-Lab/data-science.git data_science
```

## Data

### titanic

```python
import seaborn as sns
titanic = sns.load_dataset("titanic")
titanic.to_csv("data_science/data/titanic.csv", index=False)
```
