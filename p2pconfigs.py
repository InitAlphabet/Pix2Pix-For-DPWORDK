import tool

# 配置文件
# 配置设置
train_configs = tool.ConfigDict({
    "TRAIN_DATA_PATH": "data/GAN/train_A",  # 训练集位置
    "TRAIN_TARGET_PATH": "data/GAN/train_B",  # 训练集标准文件
    "TEST_DATA_PATH": "data/GAN/test_A",  # 测试集位置
    "TEST_TARGET_PATH": "data/GAN/test_B",  # 测试集的标准文件
    "IMAGE_SIZE": 256,  # 图片大小
    "EPOCHS": 100,  # 训练伦次
    "VISUALIZE_EVERY": 5,  # 每多少轮进行可视化测试
    "G_LR": 2e-4,  # 生成器学习率
    "D_LR": 2e-4,  # 判别器学习率
    "BATCH_SIZE": 16,  # 批大小
    "LAMBDA_L1": 100,  # L1损失 缩放参数
    "G_BETA": (0.5, 0.999),  # 生成器的adam优化器参数
    "D_BETA": (0.5, 0.999),  # 判别器的adam优化器参数
    "EXIST_MODEL_PT": False,  # 是否有预训练的权重
    "MODEL_PT_PATH": ""  # EXIST_MODEL_PT 为true时，此处填入模型权重位置
})

test_configs = tool.ConfigDict({
    "MODEL_TYPE": 'common',  # 使用的生成模型
    "MODEL_PT_PATH": "output/train/train1/model/checkpoint_epoch_1.pth",  # 训练好的权重文件
    "TEST_DATA_PATH": "data/GAN/test_A",  # 测试集位置
    "TEST_OUTPUT_PATH": "output/test/test1",  # 测试输出文件夹
    "TEST_TARGET_PATH": "data/GAN/test_B",  # 测试集的标准文件
    "IMAGE_SIZE": 256,  # 图片大小
})
