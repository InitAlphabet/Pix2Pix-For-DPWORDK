import os
from pathlib import Path
import re


class ConfigDict(dict):
    # 编码配置文件
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        # 返回字典中对应的值
        if item in self:
            return self[item]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


class ParameterError(Exception):
    pass


def create_unique_directory(base_dir, prefix="train"):
    """
    在指定的 base_dir 目录下创建唯一的子目录，目录名称为 train_x，
    x 是现有目录中最大的编号 + 1，确保没有重复。

    :param base_dir: 基本目录，子目录将在该目录下创建
    :param prefix: 子目录的前缀，默认是 "train"
    :return: 创建的子目录路径
    """
    # 确保 base_dir 存在
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # 获取当前目录下所有符合 trainx 格式的文件夹
    existing_dirs = [d for d in os.listdir(base_dir) if re.match(rf"^{prefix}(\d+)$", d)]

    # 提取出所有符合的目录编号
    numbers = [int(re.match(rf"^{prefix}(\d+)$", d).group(1)) for d in existing_dirs]

    # 如果没有符合的目录，则创建 train_1，否则创建最大编号 + 1
    new_dir_number = max(numbers, default=0) + 1
    new_dir = os.path.join(base_dir, f"{prefix}{new_dir_number}")

    # 创建新的目录
    os.mkdir(new_dir)

    return new_dir
