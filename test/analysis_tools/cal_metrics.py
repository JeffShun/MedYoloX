"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser
import os
import numpy as np
from runstats import Statistics
import pandas as pd

def iou(datas):
    pred, label = datas["pred"], datas["label"]
    intersection = (pred*label).sum()
    union = pred.sum() + label.sum() - intersection
    IOU = intersection / union
    return IOU


def precision(datas):
    iou_threshold = datas["iou_threshold"]
    IOU = iou(datas)
    if IOU > iou_threshold:
        return 1
    else:
        return 0

class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metric_funcs = metric_funcs
        self.metrics = {metric: Statistics() for metric in metric_funcs}
        self.metrics_data = {metric:[] for metric in metric_funcs}

    def push(self, push_data):
        for metric, func in self.metric_funcs.items():
            val = func(push_data)
            self.metrics[metric].push(val)
            self.metrics_data[metric].append((push_data["pid"], val))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}
    

    def save(self, save_dir):
        os.makedirs(save_dir,exist_ok=True)
        df = pd.DataFrame()
        # 遍历数据字典，将每种评价方式的数据添加为DataFrame的一列
        for method, values in self.metrics_data.items():
            labels, scores = zip(*values)
            df[method] = scores
        df['pid'] = labels
        df = df[['pid'] + [col for col in df.columns if col != 'pid']]
        csv_file_path = save_dir / "metrics.csv"
        df.to_csv(csv_file_path, index=False)


    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )

def evaluate(args):
    METRIC_FUNCS = dict(
        PRECISION=precision,
        IOU=iou
        )    
    metrics = Metrics(METRIC_FUNCS)
    iou_threshold = args.iou_threshold
    for sample in args.data_path.iterdir():
        if not str(sample).endswith("npz"):
            continue
        pid = str(sample).split("\\")[-1].replace("npz","")
        data = np.load(sample, allow_pickle=True)
        pred = data['pred']
        label = data['label']
        push_data = {
            "pid": pid,
            "pred": pred,
            "label": label,
            "iou_threshold": iou_threshold
        }
        metrics.push(push_data)

    return metrics


if __name__ == "__main__":
    
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data_path", type=pathlib.Path,
        default=pathlib.Path("./data/output/liver/data_for_metrics"),
    )
    parser.add_argument(
        "--iou_threshold", type=float,
        default=0.5,
    )
    args = parser.parse_args()
    metrics = evaluate(args)
    metrics.save(args.data_path)
    print(metrics)
