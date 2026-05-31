import argparse


parser = argparse.ArgumentParser()
print("parser object:", parser)

parser.add_argument("--model")
parser.add_argument("--input")
parser.add_argument("--output")

print("parser object:", parser)

args = parser.parse_args()
print("#" * 40)
print("args:", args.MODEL)
