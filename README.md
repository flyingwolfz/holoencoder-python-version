# holoencoder-python-version

unofficial python version of holoencoder (from "High-speed computer-generated holography  using an autoencoder-based deep neural network"). pytorch is used.

the Unet is almost same as the official matlab version from THUHoloLab(https://github.com/THUHoloLab/Holo-encoder)

Instead, we use band-limited ASM （from"Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields"）. And the loss function is MSE.

PSNR and SSIM is a little higher than original paper.It is tested using RTX3080， DIV2K validation dataset.


非官方python版holoencoder(来自"High-speed computer-generated holography  using an autoencoder-based deep neural network").使用pytorch.

Unet部分与THUHoloLab官方的matlab版几乎相同（https://github.com/THUHoloLab/Holo-encoder ）。
 
衍射计算使用带限角谱（来自"Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields"）。损失函数是MSE。

PSNR和SSIM比原来的论文高一点。测试指标使用RTX3080， DIV2K validation dataset


![捕获](https://user-images.githubusercontent.com/57349703/175008808-c254a22e-359e-480c-a9f9-ec03a64c3172.PNG)
