```bash
!rm -rf data
!git clone https://github.com/GLI-Lab/data.git data
```

```python
import sys
sys.path.append('/content/data')

import numpy as np
from MUTAG.loader import MUTAGFromCSV


dataset = MUTAGFromCSV()

print(f"Total {len(dataset)} Graphs: {dataset.data}")
print(f"Total Label Distribution: {np.bincount(dataset.y)}")
print(f"Graph (1/{len(dataset)}): {dataset[0]}")
# print(f"- Node Features {list(dataset[0].x.shape)}:\n{dataset[0].x}")
# print(f"- Edge Features {list(dataset[0].edge_attr.shape)}:\n{dataset[0].edge_attr}")
```
