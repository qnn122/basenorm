import argparse
from src.dataset import load_bb4_data
from src.trainer import BB4NormTrainer
from src.utils import accuracy

def main(args):
    ################################################
    print("\nLOADING DATA:\n")
    ################################################
    dd_ref, dd_train, dd_test, dd_habDev = load_bb4_data('BB4')

    trainer = BB4NormTrainer(args, train_dataset=dd_train, ref_dataset=dd_ref, test_dataset=dd_test)

    if args.do_train:
        trainer.train()

        dd_predictions = trainer.inference()
        print("Evaluating BB4 results on BB4 dev...")
        score_BB4_onDev = accuracy(dd_predictions, dd_habDev)
        print("score_BB4_onDev:", score_BB4_onDev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.1",
        help="Model Name or Path",
    )

    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--embbed_size",
        default=768,
        type=int,
        help="Embedding size",
    )

    parser.add_argument(
        "--max_length",
        default=40,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()

    main(args)
