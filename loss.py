import torch


class Loss:
    """
    损失函数类
    """
    @staticmethod
    def generator_loss(disc_fake_output, gen_output, target, lambda_adv=1):
        adversarial_loss = torch.mean((disc_fake_output - 1) ** 2)  # LSGAN loss
        l1_loss = torch.mean(torch.abs(target - gen_output))
        # print(f"GEN: adv:{lambda_adv*adversarial_loss.item()}, l1:{ l1_loss.item()}")
        return lambda_adv * adversarial_loss, l1_loss

    @staticmethod
    def discriminator_loss(disc_real_output, disc_fake_output):
        real_loss = torch.mean((disc_real_output - 1) ** 2)  # 促进真品识别
        fake_loss = torch.mean(disc_fake_output ** 2)  # 抑制仿造品
        # print(f"DIS: fake:{fake_loss.item()}, real:{real_loss.item()}")
        return real_loss, fake_loss
