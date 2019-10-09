import numpy as np
import pandas as pd

input_batch_tokens = np.random.randn(30, 158)
input_get = []

for one_row in input_batch_tokens:
    input_get.append(",".join([str(k) for k in one_row]))

target_batch = np.random.randint(0, 4, (30), dtype=np.int32)

print(target_batch)

csv_data = pd.DataFrame(
    {'vec': input_get,
     'label': target_batch}
)
csv_data.to_csv('./data.csv')
