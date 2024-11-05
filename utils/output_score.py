import argparse
import os
import json


def output_score(dir, score_file_name="avg_scores.json", only_avg=True):
    print(f"Output score for {dir}")

    if not os.path.exists(dir):
        print(f"{dir} does not exist.")
        return

    for file in os.listdir(dir):
        e_name = file.split(".")[0]

        score_file = os.path.join(dir, file, score_file_name)

        if os.path.exists(score_file):
            with open(score_file, "r") as f:
                scores = json.load(f)
                eval_time = scores.get("eval_time", "N/A")
                avg_scores = scores.get("avg_scores", "N/A")
                if only_avg:
                    avg_scores = avg_scores.get("avg", "N/A")

                print(f"{e_name}: {avg_scores} (evaluated at {eval_time})")
        else:
            print(f"{score_file} does not exist.")




if __name__ == "__main__":
    # 接受一个dir参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--score_file", type=str, default="avg_scores.json")
    parser.add_argument("--only_avg", type=bool, default=True)
    args = parser.parse_args()

    output_score(args.dir)