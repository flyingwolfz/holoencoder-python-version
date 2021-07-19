# holoencoder-python-version
python version of holoencoder (from "High-speed computer-generated holography  using an autoencoder-based deep neural network"). pytorch is used.

the Unet is same as the matlab version from THUHoloLab(https://github.com/THUHoloLab/Holo-encoder)

Instead, we use band-limited ASM （from"Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields"）. And the loss function is MSE.

Pretrained model is included, you can test the network in "loadmodel.py".

The result is far from satisfaction, although it looks fine when the image is small.


The pretrained network uses 1024*1024 images in 100 loops.Larger input image and more training loop can make result better. 

python版holoencoder(来自"High-speed computer-generated holography  using an autoencoder-based deep neural network").使用pytorch.

Unet部分与THUHoloLab的matlab版相同（https://github.com/THUHoloLab/Holo-encoder ）。
 
衍射计算使用带限角谱（来自"Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields"）。损失函数是MSE。

包含已经训练好的模型，在“loadmodel.py”中可以测试。

效果还差很多，因为成像很小，所以成像看起来还行。

预训练模型是用1024*1024分辨率训练了100循环。用更高分辨率图片训练更多轮，效果会好很多。
