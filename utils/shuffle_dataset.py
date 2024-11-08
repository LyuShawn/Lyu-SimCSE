import argparse
import os
import random

def shuffle_dataset(dataset, output, seed=0):
    if not os.path.exists(dataset):
        print(f"{dataset} does not exist.")
        return

    print(f"Shuffle dataset {dataset} to {output}")

    if seed:
        random.seed(seed)
    else:
        random.seed()

    with open(dataset, "r") as f:
        lines = f.readlines()

    random.shuffle(lines)

    with open(output, "w") as f:
        f.writelines(lines)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shuffle_dataset(args.dataset, args.output, args.seed)