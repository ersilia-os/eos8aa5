import os
import sys
import csv
import numpy as np

data_dir = sys.argv[1]
output_file = sys.argv[2]

embeddings_file = os.path.join(data_dir, "data", "kpgt_base.npz")
data = np.load(embeddings_file)
X = data["fps"]

col_idxs = [i for i in range(X.shape[1])]
col_names = ["feat-{0}".format(str(idx).zfill(4)) for idx in col_idxs]

with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)
    for i in range(X.shape[0]):
        writer.writerow(list(X[i,:]))
