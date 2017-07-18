import os

from argparse import ArgumentParser
from basic.main import main as m

def get_args():
    parser = ArgumentParser(description='PyTorch BIDAF model')

    # Names and directories
    parser.add_argument("--model_name", type=str, default="basic", help="Model name [basic]")
    parser.add_argument("--data_dir", type=str, default="data/squad", help="Data dir [data/squad]")
    parser.add_argument("--run_id", type=str, default="0", help="Run ID [0]")
    parser.add_argument("--out_base_dir", type=str, default="out", help="out base dir [out]")
    parser.add_argument("--forward_name", type=str, default="single", help="Forward name [single]")
    parser.add_argument("--answer_path", type=str, default="", help="Answer path []")
    parser.add_argument("--eval_path", type=str, default="", help="Eval path []")
    parser.add_argument("--load_path", type=str, default="", help="Load path []")
    parser.add_argument("--shared_path", type=str, default="", help="Shared path []")

    # Essential training and test options
    parser.add_argument("--mode", type=str, default="train", help="train | test | forward [test]")
    parser.add_argument("--load", type=bool, default=True, help="load saved data? [True]")
    parser.add_argument("--single", type=bool, default=False, help="supervise only the answer sentence? [False]")
    parser.add_argument("--debug", type=bool, default=False, help="Debugging mode? [False]")
    parser.add_argument('--load_ema', type=bool, default=True, help="load exponential average of variables when testing?  [True]")
    parser.add_argument("--eval", type=bool, default=True, help="eval? [True]")
    parser.add_argument("--wy", type=bool, default=False, help="Use wy for loss / eval? [False]")
    parser.add_argument("--na", type=bool, default=False, help="Enable no answer strategy and learn bias? [False]")
    parser.add_argument("--th", type=float, default=0.5, help="Threshold [0.5]")

    # Training / test parameters
    parser.add_argument("--batch_size", type=int, default=60, help="Batch size [60]")
    parser.add_argument("--val_num_batches", type=int, default=100, help="validation num batches [100]")
    parser.add_argument("--test_num_batches", type=int, default=0, help="test num batches [0]")
    parser.add_argument("--num_epochs", type=int, default=12, help="Total number of epochs for training [12]")
    parser.add_argument("--num_steps", type=int, default=20000, help="Number of steps [20000]")
    parser.add_argument("--load_step", type=int, default=0, help="load step [0]")
    parser.add_argument("--init_lr", type=float, default=0.001, help="Initial learning rate [0.001]")
    parser.add_argument("--input_keep_prob", type=float, default=0.8, help="Input keep prob for the dropout of LSTM weights [0.8]")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Keep prob for the dropout of Char-CNN weights [0.8]")
    parser.add_argument("--wd", type=float, default=0.0, help="L2 weight decay for regularization [0.0]")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden size [100]")
    parser.add_argument("--char_out_size", type=int, default=100, help="char-level word embedding size [100]")
    parser.add_argument("--char_emb_size", type=int, default=8, help="Char emb size [8]")
    parser.add_argument("--out_channel_dims", type=int, default=100, help="Out channel dims of Char-CNN, separated by commas [100]")
    parser.add_argument("--filter_heights", type=int, default=5, help="Filter heights of Char-CNN, separated by commas [5]")
    parser.add_argument("--finetune", type=bool, default=False, help="Finetune word embeddings? [False]")
    parser.add_argument("--highway", type=bool, default=True, help="Use highway? [True]")
    parser.add_argument("--highway_num_layers", type=int, default=2, help="highway num layers [2]")
    parser.add_argument("--share_cnn_weights", type=bool, default=True, help="Share Char-CNN weights [True]")
    parser.add_argument("--share_lstm_weights", type=bool, default=True, help="Share pre-processing (phrase-level) LSTM weights [True]")
    parser.add_argument("--var_decay", type=float, default=0.999, help="Exponential moving average decay for variables [0.999]")

    # Optimizations
    parser.add_argument("--cluster", type=bool, default=False, help="Cluster data for faster training [False]")
    parser.add_argument("--len_opt", type=bool, default=False, help="Length optimization? [False]")
    parser.add_argument("--cpu_opt", type=bool, default=False, help="CPU optimization? GPU computation can be slower [False]")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU needed [1]")

    # Logging and saving options
    parser.add_argument("--progress", type=bool, default=True, help="Show progress? [True]")
    parser.add_argument("--log_period",  type=int, default=100, help="Log period [100]")
    parser.add_argument("--eval_period", type=int, default= 1000, help="Eval period [1000]")
    parser.add_argument("--save_period", type=int, default= 1000, help="Save Period [1000]")
    parser.add_argument("--max_to_keep", type=int, default= 20, help="Max recent saves to keep [20]")
    parser.add_argument("--dump_eval", type=bool, default=True, help="dump eval? [True]")
    parser.add_argument("--dump_answer", type=bool, default=True, help="dump answer? [True]")
    parser.add_argument("--vis", type=bool, default=False, help="output visualization numbers? [False]")
    parser.add_argument("--dump_pickle", type=bool, default=True, help="Dump pickle instead of json? [True]")
    parser.add_argument("--decay", type=float, default=0.9, help="Exponential moving average decay for logging values [0.9]")

    # Thresholds for speed and less memory usage
    parser.add_argument("--word_count_th", type=float, default=10, help="word count th [100]")
    parser.add_argument("--char_count_th", type=float, default=50, help="char count th [500]")
    parser.add_argument("--sent_size_th", type=float, default=400, help="sent size th [64]")
    parser.add_argument("--num_sents_th", type=float, default=8, help="num sents th [8]")
    parser.add_argument("--ques_size_th", type=float, default=30, help="ques size th [32]")
    parser.add_argument("--word_size_th", type=float, default=16, help="word size th [16]")
    parser.add_argument("--para_size_th", type=float, default=256, help="para size th [256]")

    # Advanced training options
    parser.add_argument("--lower_word", type=bool, default=True, help="lower word [True]")
    parser.add_argument("--squash", type=bool, default=False, help="squash the sentences into one? [False]")
    parser.add_argument("--swap_memory", type=bool, default=True, help="swap memory? [True]")
    parser.add_argument("--data_filter", type=str, default="max", help="max | valid | semi [max]")
    parser.add_argument("--use_glove_for_unk", type=bool, default=True, help="use glove for unk [False]")
    parser.add_argument("--known_if_glove", type=bool, default=True, help="consider as known if present in glove [False]")
    parser.add_argument("--logit_func", type=str, default="tri_linear", help="logit func [tri_linear]")
    parser.add_argument("--answer_func", type=str, default="linear", help="answer logit func [linear]")
    parser.add_argument("--sh_logit_func", type=str, default="tri_linear", help="sh logit func [tri_linear]")

    # Ablation options
    parser.add_argument("--use_char_emb", type=bool, default=True, help="use char emb? [True]")
    parser.add_argument("--use_word_emb", type=bool, default=True, help="use word embedding? [True]")
    parser.add_argument("--q2c_att", type=bool, default=True, help="question-to-context attention? [True]")
    parser.add_argument("--c2q_att", type=bool, default=True, help="context-to-question attention? [True]")

    args = parser.parse_args()
    return args

def main():
    config = get_args()
    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))
    print("In main.....")

    m(config)

if __name__ == "__main__":
    main()
