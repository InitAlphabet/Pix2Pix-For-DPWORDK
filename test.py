import glob
import os
import torch
from PIL import Image
from torchvision.transforms import transforms

from pix2pix import Generator
from p2pconfigs import test_configs as configs

model_path = configs.MODEL_PT_PATH  # 模型位置
output_dir = configs.TEST_OUTPUT_PATH  # 测试输出位置
target_dir = configs.TEST_TARGET_PATH  # 标准输出的为止
os.makedirs(output_dir, exist_ok=True)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator(configs.MODEL_TYPE)
pts = torch.load(model_path, weights_only=True)
model.load_state_dict(pts['generator_state_dict'])
model.to(device)
model.eval()

# 数据预处理操作
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化调整特征尺度。
])


# 生成并保存图像的函数
def generate_and_save_images(input_image_paths):
    os.makedirs(output_dir, exist_ok=True)
    generated_img_paths = []
    for i, input_image_path in enumerate(input_image_paths):
        img = Image.open(input_image_path).convert("RGB")
        img_tensor = transformer(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            gen_output = model(img_tensor)
            gen_output = gen_output.squeeze(0)
            gen_output = gen_output.cpu().numpy().transpose(1, 2, 0)
            gen_output = (gen_output + 1) * 127.5
            gen_output = gen_output.astype('uint8')
            gen_output = Image.fromarray(gen_output)
        # 保存生成的图像
        generated_image_path = os.path.join(output_dir, f"{i}.png")
        gen_output.save(generated_image_path)
        generated_img_paths.append(generated_image_path)

    return generated_img_paths


target_png_paths = sorted(glob.glob((target_dir + "/*.png")))
generate_and_save_images(target_png_paths)
