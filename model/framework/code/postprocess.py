import os
import sys
import csv
import numpy as np
import json

data_dir = sys.argv[1]
output_file = sys.argv[2]

embeddings_file = os.path.join(data_dir, "data", "kpgt_base.npz")
data = np.load(embeddings_file)
X = data["fps"]

col_idxs = [i for i in range(X.shape[1])]
header = ["dim_{0}".format(str(idx).zfill(4)) for idx in col_idxs]
outputs = [list(X[i, :]) for i in range(X.shape[0])]

def write_out_csv(results, header, file):
  with open(file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for r in results:
      writer.writerow(r)

def write_out_bin(results, header, file):
  arr = np.asarray(results, dtype=np.float32)
  meta = {"columns": header, "shape": arr.shape, "dtype": "float32"}
  meta_bytes = (json.dumps(meta) + "\n").encode("utf-8")
  with open(file, "wb") as f:
    f.write(meta_bytes)
    f.truncate(len(meta_bytes) + arr.nbytes)
  m = np.memmap(
    file, dtype=arr.dtype, mode="r+", offset=len(meta_bytes), shape=arr.shape
  )
  m[:] = arr
  m.flush()

def write_out(results, header, file):
  if file.endswith(".bin"):
    write_out_bin(results, header, file)
  elif file.endswith(".csv"):
    write_out_csv(results, header, file)
  else:
    raise ValueError(f"Unsupported extension for {file!r}")

write_out(outputs, header, output_file)
