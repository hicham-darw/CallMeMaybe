import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--input", default="../../data/input/...")
parser.add_argument("--output", default="../../data/output/...")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
args = parser.parse_args()
print("Args: ", args.input)
print("Args: ", args.output)
print("Args: ", args.model)


# parser = argparse.ArgumentParser()
# print("parser object:", parser)

# input_path = ""
# output_path = ""
# model = ""
# parser.add_argument("--model", help="call specific model", dest='model')
# parser.add_argument("--input", help="override input path", dest="input")
# parser.add_argument("--output", help='override output path', dest="output")

# print("parser object:", parser)
# args = parser.parse_args()
# print("input:", args.input)
# parser = argparse.ArgumentParser()
# print("parser object:", parser)

# input_path = ""
# output_path = ""
# model = ""
# parser.add_argument("--model", help="call specific model", dest='model')
# parser.add_argument("--input", help="override input path", dest="input")
# parser.add_argument("--output", help='override output path', dest="output")

# print("parser object:", parser)
# args = parser.parse_args()
# print("input:", args.input)