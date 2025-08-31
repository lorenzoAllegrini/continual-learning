import os

import pandas as pd
from spaceai.data import NASA
from spaceai.benchmark import NASABenchmark
from spaceai.models.anomaly import Telemanom
from spaceai.models.predictors import PNN

import torch
from torch import nn
from torch import optim

from spaceai.benchmark.callbacks import SystemMonitorCallback
from avalanche.training.supervised import Naive
from spaceai.benchmark.utils import CLTrainer
from spaceai.data.utils import seq_collate_fn

def main():
    
    benchmark = NASABenchmark(
        run_id="nasa_pnn_lstm_2",
        exp_dir="experiments",
        seq_length=250,
        n_predictions=1,
        data_root="datasets",
    )
    callbacks = [SystemMonitorCallback()]

    for sensor_channels in [NASA.smap_channels, NASA.msl_channels]:
        groups = set([el.split("-")[0] for el in sensor_channels])
        for i, group in enumerate(groups):
            group_channels = [ch for ch in sensor_channels if ch.startswith(group)]
            print(f"{i+1}/{len(group_channels)}: {group}")
            sample_channel = NASA("datasets", group_channels[0], mode="anomaly", train=False)

            predictor = PNN(
                num_layers=2,
                in_features=sample_channel.in_features_size,
                hidden_features_per_column=80,
                adapter='mlp',
                base_predictor_args=dict(
                    hidden_sizes=[80],
                    dropout=0.3,
                    washout=249,
                ),      
            )
            
            for i, channel_id in enumerate(group_channels):

                detector = Telemanom(
                    pruning_factor=0.13, force_early_anomaly=channel_id == "C-2"
                )

                criterion = nn.MSELoss()
                optimizer_factory = lambda predictor: optim.Adam(predictor.parameters(), lr=0.001)
                epochs = 35

                # definizione della strategia naive
                trainer = CLTrainer(model=predictor, optimizer_factory=optimizer_factory,
                                criterion=criterion, device="cuda" if torch.cuda.is_available() else "cpu",
                                collate_fn=seq_collate_fn(n_inputs=2, mode="time"), train_epochs=epochs,
                                patience_before_stopping=3, min_delta=0.0)

                benchmark.run_pnn(
                    channel_id,
                    predictor,
                    detector,
                    strategy=trainer,
                    overlapping_train=True,
                    restore_predictor=False,
                    callbacks=callbacks,
                    perc_eval=0.2
                )

    results_df = pd.read_csv(os.path.join(benchmark.run_dir, "results.csv"))
    tp = results_df["true_positives"].sum()
    fp = results_df["false_positives"].sum()
    fn = results_df["false_negatives"].sum()

    total_precision = tp / (tp + fp)
    total_recall = tp / (tp + fn)
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)

    print("True Positives: ", tp)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)
    print("Total Precision: ", total_precision)
    print("Total Recall: ", total_recall)
    print("Total F1: ", total_f1)


if __name__ == "__main__":
    main()
