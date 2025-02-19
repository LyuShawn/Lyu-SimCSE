import amrlib
from tqdm import tqdm
import argparse
import torch
import os

def main(args):
    input_file = args.input_file
    output_file = args.output_file
    bs = args.bs

    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"Loading sents...")
    with open(input_file, 'r', encoding='utf-8') as f:
        sent_list = f.read().splitlines()

    print(f"Loading models...")
    stog = amrlib.load_stog_model(args.stog_model)
    stog.max_sent_len = 64
    stog.max_graph_len = 256
    stog.batch_size = bs
    gtos = amrlib.load_gtos_model("model/model_generate_t5wtense-v0_1_0")
    gtos.batch_size = bs
    gtos.max_sent_len = 64
    gtos.max_graph_len = 256
    print(f"Generating AMRs...")
    for i in tqdm(range(0, len(sent_list), bs)):
        list = sent_list[i:i+bs]
        graphs = stog.parse_sents(list)
        sents = gtos.generate(graphs)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(sents[0]) + '\n')
        # 清理显存
        del graphs
        del sents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='data/wiki1m_for_simcse.txt')
    parser.add_argument('--output_file', type=str, default='data/wiki1m_for_simcse_amr.txt')
    parser.add_argument('--stog_model', type=str, default='model/model_parse_xfm_bart_base-v0_1_0')
    parser.add_argument('--bs', type=int, default=16)
    args = parser.parse_args()
    main(args)