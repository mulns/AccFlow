import os

import lmdb
import pyarrow as pa
from tqdm import tqdm

file = "data/datasets/CVO_full/cvo_train.lmdb"
env = lmdb.open(
    file,
    subdir=os.path.isdir(file),
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)
txn = env.begin(write=False)
print(txn.stat())
# for k, _ in tqdm(txn.cursor()):
#     key = str(k)
#     if "flows" in key or "imgs" in key:
#         continue
#     else:
#         print(key)


with env.begin(write=False) as txn:
    all_keys = pa.deserialize(txn.get(b"__keys__"))
    samples = pa.deserialize(txn.get(b"__samples__"))
    keys = pa.deserialize(txn.get(b"__valid_keys__"))
    length = len(samples)
print(all_keys)
print(samples)
print(keys)

# print("------------------")


# print("------------------")

# print(keys)

# print("------------------")
