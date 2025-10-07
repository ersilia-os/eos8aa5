# imports
import os
import csv
import sys
import json
import struct

# parse arguments
input_file = sys.argv[1]
output_dir = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

# functions to read and write .csv and .bin files
def read_smiles_csv(in_file): # read SMILES from .csv file, assuming one column with header
  with open(in_file, "r") as f:
    reader = csv.reader(f)
    cols = next(reader)
    data = [r[0] for r in reader]
    return cols, data

def read_smiles_bin(in_file):
  with open(in_file, "rb") as f:
    data = f.read()

  mv = memoryview(data)
  nl = mv.tobytes().find(b"\n")
  meta = json.loads(mv[:nl].tobytes().decode("utf-8"))
  cols = meta.get("columns", [])
  count = meta.get("count", 0)
  smiles_list = [None] * count
  offset = nl + 1
  for i in range(count):
    (length,) = struct.unpack_from(">I", mv, offset)
    offset += 4
    smiles_list[i] = mv[offset : offset + length].tobytes().decode("utf-8")
    offset += length
  return cols, smiles_list

def read_smiles(in_file):
  if in_file.endswith(".bin"):
    return read_smiles_bin(in_file)
  return read_smiles_csv(in_file)

# read input
_, smiles_list = read_smiles(input_file)

# create output file
output_file = os.path.join(output_dir, "data", "data.csv")
os.mkdir(os.path.join(output_dir, "data"))
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for smiles in smiles_list:
        writer.writerow([smiles])
