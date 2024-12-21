from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import os

import tool
from p2pconfigs import train_configs as configs
from loss import Loss
from DataManager import Pix2PixDataset
from pix2pix import Discriminator, Generator
# torch版本 2.0.0+cu118,python==3.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理操作
transformer = transforms.Compose([
    transforms.Resize((configs.IMAGE_SIZE, configs.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化调整特征尺度。
])

# 数据加载
train_dataset = Pix2PixDataset(configs.TRAIN_DATA_PATH, configs.TRAIN_TARGET_PATH, transformer)
test_dataset = Pix2PixDataset(configs.TEST_DATA_PATH, configs.TEST_TARGET_PATH, transformer)

# 数据转换
train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)


# 初始化模型和优化器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=configs.G_LR, betas=configs.G_BETA)
optimizer_D = optim.Adam(discriminator.parameters(), lr=configs.D_LR, betas=configs.D_BETA)


# 保存模型函数
def save_model(**kwargs):
    """
    :param kwargs: checkpoint_dir, epochs
    :return:
    """
    __checkpoint_dir = kwargs['checkpoint_dir']
    __epochs = kwargs['epochs']
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'epoch': __epochs
    }
    torch.save(checkpoint, f"{__checkpoint_dir}/checkpoint_epoch_{__epochs}.pth")
    print(f"Model saved at epoch {__epochs}")


# 加载模型函数
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    __start_epoch = checkpoint['epoch']
    print(f"Model loaded from {checkpoint_path}, starting from epoch {__start_epoch}")
    return __start_epoch


def draw_loss(loss_dict, end_epoch):
    epochs = list(range(1, end_epoch + 1))

    # 损失可视化
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, loss_dict['L1_losses'], label='L1', color='blue', marker='o')
    plt.title('L1_LOSSES')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{loss_dict['output_dir']}/L1_LOSS.png")

    plt.figure(figsize=(12, 8))
    # 绘制每个损失
    plt.plot(epochs, loss_dict['G_adv_losses'], label='ADV Loss', color='blue')
    plt.plot(epochs, loss_dict['D_fake_losses'], label='Fake Loss', color='red')
    plt.plot(epochs, loss_dict['D_real_losses'], label='Real Loss', color='green')

    # 添加标签和标题
    plt.xlabel('Epochs')
    plt.ylabel('Loss ')
    plt.title('Losses during Training')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{loss_dict['output_dir']}/GAN_LOSS.png")


# 创建可视化输出文件夹
output_dir = tool.create_unique_directory("output/train", "train")
model_save_dir = output_dir + "/model"
epoch_save_dir = output_dir + "/epochs"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(epoch_save_dir, exist_ok=True)
G_adv_losses, L1_losses, D_fake_losses, D_real_losses = [], [], [], []
out_dict = {"output_dir": output_dir}
# 开始训练
start_epoch = 0

for epoch in range(start_epoch, start_epoch + configs.EPOCHS):
    generator.train()
    adv_loss, l1_loss, real_loss, fake_loss = 0, 0, 0, 0
    for img, target in train_loader:
        img, target = img.to(device), target.to(device)

        optimizer_G.zero_grad()
        gen_output = generator(img)
        disc_fake_output = discriminator(img, gen_output)
        adv_loss, l1_loss = Loss.generator_loss(disc_fake_output, gen_output, target)
        gen_loss = adv_loss + l1_loss * configs.LAMBDA_L1
        gen_loss.backward()
        optimizer_G.step()
        optimizer_D.zero_grad()
        disc_real_output = discriminator(img, target)
        disc_fake_output = discriminator(img, gen_output.detach())
        real_loss, fake_loss = Loss.discriminator_loss(disc_real_output, disc_fake_output)
        disc_loss = (real_loss + fake_loss) * 0.5
        disc_loss.backward()
        optimizer_D.step()
    G_adv_losses.append(adv_loss.item())
    L1_losses.append(l1_loss.item())
    D_fake_losses.append(fake_loss.item())
    D_real_losses.append(real_loss.item())

    print(f"Epoch [{epoch + 1}/{start_epoch + configs.EPOCHS}], "
          f" L1*LAMBDA:{configs.LAMBDA_L1 * l1_loss.item()},GEN_L1:{l1_loss.item()}, GEN_ADV:{adv_loss.item():.4f} ,"
          f" D_FAKE:{fake_loss.item():.4f}, D_REAL:{real_loss.item():.4f}")

    # 可视化生成结果
    if (epoch + 1) % configs.VISUALIZE_EVERY == 0:
        generator.eval()
        with torch.no_grad():
            img, target = next(iter(test_loader))
            img, target = img.to(device), target.to(device)
            gen_output = generator(img)

        # 绘制图像
        fig, ax = plt.subplots(3, 4, figsize=(12, 9))
        for i in range(4):
            # 转置matplotlib格式，进行还原图像。
            ax[0, i].imshow(((img[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8'))
            ax[1, i].imshow(((target[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8'))
            ax[2, i].imshow(((gen_output[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8'))
            ax[0, i].set_title("Input")
            ax[1, i].set_title("Target")
            ax[2, i].set_title("Generated")
            for j in range(3):
                ax[j, i].axis('off')

        plt.tight_layout()
        plt.savefig(f"{epoch_save_dir}/epoch_{epoch + 1}.png")
        plt.close(fig)

out_dict["G_adv_losses"] = G_adv_losses
out_dict["D_real_losses"] = D_real_losses
out_dict["L1_losses"] = L1_losses
out_dict['D_fake_losses'] = D_fake_losses
draw_loss(loss_dict=out_dict, end_epoch=configs.EPOCHS)
save_model(checkpoint_dir=model_save_dir, epochs=start_epoch + configs.EPOCHS)
