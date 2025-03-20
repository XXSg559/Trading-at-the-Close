import pandas as pd
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


def load_and_process_data(file_path="optiver-trading-at-the-close/train.csv"):

    data = pd.read_csv(file_path)

    # 用 far_price 的中位数 填充对应列的nan
    data["far_price"] = data["far_price"].fillna(data["far_price"].median())
    # 用 near_price 的中位数 填充
    data["near_price"] = data["near_price"].fillna(data["near_price"].median())

    # 删除缺失行 220条
    data = data.dropna()
    # 去掉最后两列
    data = data.drop(columns=["time_id", "row_id"])

    # 划分数据集，共481天，后80天为测试集，前321天为训练集，中间80天为验证集
    # 能用torch的dataloader吗
    train_data = data[data["date_id"] <= 320]
    valid_data = data[(data["date_id"] > 320) & (data["date_id"] < 400)]
    test_data = data[data["date_id"] > 400]

    return data


class StockDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length
        self.groups = data.groupby("stock_id")
        self.obs_per_day = 55
        self.total_obs = (seq_length + 1) * self.obs_per_day

        self.samples = []
        for stock, stock_data in self.groups:
            num_days = len(stock_data) // self.obs_per_day
            for i in range(num_days - seq_length):
                start_idx = i * self.obs_per_day
                end_idx = (i + seq_length + 1) * self.obs_per_day
                sample_data = stock_data.iloc[start_idx:end_idx]
                self.samples.append(sample_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        columns_to_drop = ["stock_id", "date_id", "seconds_in_bucket"]
        sample = sample.drop(columns=columns_to_drop)
        target = sample.iloc[-self.obs_per_day :][["target"]]
        input_seq = sample.iloc[: -self.obs_per_day]

        # 检查 input_seq 的列数
        input_cols = len(input_seq.columns)

        # 补齐数据
        input_tensor = np.full((self.total_obs - self.obs_per_day, input_cols), np.nan)
        input_tensor[: len(input_seq)] = input_seq.values
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

        target_tensor = np.full((self.obs_per_day, 1), np.nan)
        target_tensor[: len(target)] = target.values
        target_tensor = torch.tensor(target_tensor, dtype=torch.float32)

        # 创建掩码
        # input_mask = ~torch.isnan(input_tensor)
        # target_mask = ~torch.isnan(target_tensor)

        input_tensor[torch.isnan(input_tensor)] = 0
        target_tensor[torch.isnan(target_tensor)] = 0

        return input_tensor  # target_tensor, input_mask, target_mask


def create_stock_dataset(data=None, seq_length=30):
    """
    创建并返回一个 StockDataset 实例。
    """
    return StockDataset(data, seq_length=30)
