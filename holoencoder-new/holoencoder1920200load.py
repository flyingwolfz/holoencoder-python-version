import torch
from torch import nn, optim
import cv2
import numpy
import torch.fft
import math
import time
from torchsummary import summary
from skimage.metrics import structural_similarity as ssim
z=200 #juli

pitch=0.008
wavelength=0.000638
n = 1072
m = 1920

x = numpy.linspace(-n//2, n//2-1, n)
y = numpy.linspace(-m//2, m//2-1, m)
x=torch.from_numpy(x)
y=torch.from_numpy(y)
n=numpy.array(n)
m=numpy.array(m)
n = torch.from_numpy(n)
m = torch.from_numpy(m)

v = 1 / (n * pitch)
u = 1 / (m * pitch)
fx = x * v
fy = y * u

fX, fY = torch.meshgrid(fx, fy)

H = (-1)*(2*numpy.pi/wavelength) * z * torch.sqrt(1 - (wavelength*fX)**2 - (wavelength*fY)**2)
Hreal=torch.cos(H)
Himage=torch.sin(H)

xlimit=1/torch.sqrt((2*1/m/pitch*z)**2+1)/wavelength
ylimit=1/torch.sqrt((2*1/n/pitch*z)**2+1)/wavelength
a = (abs(fX) < xlimit) & (abs(fY) < ylimit)
a=a.numpy()
a=a+0
Hreal=Hreal*a
Himage=Himage*a
H=torch.complex(Hreal,Himage)

H=H.cuda()

def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.net2 = nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, padding=0)
        )

    def forward(self, x):
        out1=self.net1(x)
        out2=self.skip(x)
        out3=out1+out2
        out4=self.net2(out3)
        out5=out4+out3
        return out5

class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1,output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.net2 = nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0)
        )

    def forward(self, x):
        out1=self.net1(x)
        out2=self.skip(x)
        out3=out1+out2
        out4=self.net2(out3)
        out5=out4+out3
        return out5

class holoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1=Down(1,16)
        self.netdown2=Down(16,32)
        self.netdown3=Down(32,64)
        self.netdown4=Down(64,96)
        self.netup0=Up(96,64)
        self.netup1=Up(64,32)
        self.netup2=Up(32,16)
        self.netup3=Up(16,1)
        self.tan=torch.nn.Hardtanh(-math.pi, math.pi)

    def forward(self, x):
        out1=self.netdown1(x)
        out2=self.netdown2(out1)
        out3=self.netdown3(out2)
        out4=self.netdown4(out3)

        out5=self.netup0(out4)
        out6 = self.netup1(out5+out3)
        out7 = self.netup2(out6+out2)
        out8 = self.netup3(out7+out1)
        out8=self.tan(out8)


        return out8

path='E:\\DIV2K\\DIV2K_valid_HR'
model = holoencoder()
model.load_state_dict(torch.load('holoencoderstate.pth'))

if torch.cuda.is_available():
    model.cuda()
c = 805
b = '\\0' + str(c)
imgpath = path + b + '.png'
#imgpath ='lll.png'
img = cv2.imread(imgpath)
img2 = cv2.resize(img, (1920, 1072))
gray = cv2.split(img2)[2]
#gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
cv2.imwrite('2.png', gray)
gray = numpy.reshape(gray, (1, 1, 1072, 1920))

target_amp = torch.from_numpy(gray)
target_amp = target_amp / 255.0
target_amp = target_amp.cuda()
output = model(target_amp)
summary(model.cuda(), input_size=(1, 1072,1920), batch_size=-1)
time_start=time.time()
with torch.no_grad():
 for k in range(10):
   output = model(target_amp)
   output = torch.squeeze(output)
   holo=output/2.0/3.14159+0.5

time_end=time.time()
print('totally cost',(time_end-time_start)/10.0)
holo= numpy.uint8(holo.cpu().data.numpy()*255)
cv2.imwrite('h.png', holo)
grayreal = torch.cos(output)
grayimage = torch.sin(output)
gray = torch.complex(grayreal, grayimage)
quan = torch.fft.fftn(gray)
quan2 = quan * H
final = torch.fft.ifftn(quan2)
final = torch.abs(final)
finalpic = final
target_amp=torch.squeeze(target_amp)
final = final.cpu().numpy()
target_amp = target_amp.cpu().numpy()
psnrr = psnr(final, target_amp)
print('psnr:',psnrr)
a = (finalpic>1.0)
ssimm = ssim(target_amp, final)
print('ssim:',ssimm)
#finalpic[a]=1.0
finalpic = finalpic / torch.max(finalpic)
amp = numpy.uint8(finalpic.cpu().data.numpy() * 255)
cv2.imwrite('1.png', amp)