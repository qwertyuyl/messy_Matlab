% 1.1图像的读写与显示

%%1.1.1图像的读写
%原始图片为-1
%灰度图片为-2
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
figure('name','1.1.1-1')
imshow(f)
imwrite(f,['D:\图像处理\','1.1.1-1原始图片','.png'])


%%1.1.2图像的显示
figure('name','1.1.1-2')
imshow(gf)
imwrite(gf,['D:\图像处理\','1.1.1-2灰度图片','.png'])

%1.2对原始图像进行分析

%%1.2.1原始彩色图像
whos f

%%1.2.2原始灰度图像
whos gf

%1.3图像类型转换

%%1.3.1彩色图像变成灰度图像
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
gf = rgb2gray(f);
figure('name','1.3.1')
imshow(gf)
imwrite(gf,['D:\图像处理\','1.3.1灰度图片2','.png'])

%%1.3.2灰度图像变成二值图像
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
gb = im2bw(gf, 0.4);
figure('name','1.3.2')
imshow(gb)
imwrite(gb,['D:\图像处理\','1.3.2二值图像','.png'])
%2.1亮度变换

%%2.1.1  
% 图像取反为-1  扩展灰度级为-2-3-4
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);

g1 = imadjust(gf,[0,1],[1,0]);
figure('name','2.1.1-1')
imshow(g1)
imwrite(g1,['D:\图像处理\','2.1.1-1图像取反','.png'])

g2 = imadjust(gf, [0.3, 0.55], [0,1]);
figure('name','2.1.1-2')
imshow(g2)
imwrite(g2,['D:\图像处理\','2.1.1-2扩张灰度1','.png'])

g3 = imadjust(gf, [0, 0.3], [0, 1]);
figure('name','2.1.1-3')
imshow(g3)
imwrite(g3,['D:\图像处理\','2.1.1-3扩张灰度2','.png'])

g4 = imadjust(gf, [], [], 2);
figure('name','2.1.1-4')
imshow(g4)
imwrite(g4,['D:\图像处理\','2.1.1-4扩张灰度3','.png'])

%%2.1.2对数和对比度拉伸变换
%对数变换为-1 对比度拉伸变换为-2-3
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
g1 = im2uint8(mat2gray(log(1+double(gf))));
figure('name','2.1.2-1')
imshow(g1)
imwrite(g1,['D:\图像处理\','2.1.2-1对数变换','.png'])

L = imadjust(gf,[],[50/255;150/255]);
J = imadjust(L,[50/255;150/255],[20/255;230/255]);
figure('name','2.1.2-2')
imshow(L)
imwrite(L,['D:\图像处理\','2.1.2-2对比度拉伸变换1','.png'])

figure('name','2.1.2-3')
imshow(J)
imwrite(J,['D:\图像处理\','2.1.2-3对比度拉伸变换2','.png'])



%2.2空间滤波

%%2.2.1线性空间滤波
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
h1 = [0 1 0;1 -8 1;1 1 1];
g1 = gf - imfilter(gf, h1);
figure('name','2.2.1-1')
imshow(g1,[])
imwrite(g1,['D:\图像处理\','2.2.1-1中心为-4','.png'])

h2 = [1 1 1;1 -8 1;1 1 1];
g2 = gf - imfilter(gf,h2);
figure('name','2.2.1-2')
imshow(g2,[])
imwrite(g2,['D:\图像处理\','2.2.1-2中心为-8','.png'])

%%2.2.2非线性空间滤波
%被椒盐噪声污染的图像为-1
%中值滤波为-2
%使用symmetric进行中值滤波为-3
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
fn = imnoise(gf,'salt & pepper',0.2);
figure('name','2.2.2-1')
imshow(fn)
imwrite(fn,['D:\图像处理\','2.2.2-1被椒盐噪声污染的图像','.png'])

gm = medfilt2(fn);
figure('name','2.2.2-2')
imshow(gm)
imwrite(gm,['D:\图像处理\','2.2.2-2中值滤波','.png'])

gms = medfilt2(fn,'symmetric');
figure('name','2.2.2-3')
imshow(gms)
imwrite(gms,['D:\图像处理\','2.2.2-3使用symmetric进行中值滤波','.png'])


%3.1低通滤波器
%滤波器为-1
%原始图像的谱为-2
%滤波后图像为-3
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
[gf,revertclass] = tofloat(gf);
PQ = paddedsize(size(gf));
[U,V] = dftuv(PQ(1),PQ(2));
D = hypot(U,V);
D0 = 0.05*PQ(2);
F = fft2(gf,PQ(1),PQ(2));
H = exp(-(D.^2)/(2*(D0^2)));
g = dftfilt(gf,H);
g = revertclass(g);
figure('name','3.1-1')
imshow(fftshift(H))
imwrite(fftshift(H),['D:\图像处理\','3.1-1低通滤波器','.png'])

figure('name','3.1-2')
imshow(log(1+abs(fftshift(F))),[])
imwrite(log(1+abs(fftshift(F))),['D:\图像处理\','3.1-2原始图像的谱','.png'])

figure('name','3.1-3')
imshow(g)
imwrite(g,['D:\图像处理\','3.1-3滤波后图像','.png'])

%%3.2高通滤波器
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
PQ = paddedsize(size(gf));
D0 = 0.05*PQ(1);
H = hpfilter("gaussian",PQ(1),PQ(2),D0);
g = dftfilt(gf,H);
figure('name','3.2')
imshow(g)
imwrite(g,['D:\图像处理\','3.2高通滤波后图片','.png'])

%3.3联合使用高频强调和直方图均衡
% 高频强调为-1 直方图均衡后为-2
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
PQ = paddedsize(size(gf));
D0 = 0.05*PQ(1);
HBW = hpfilter('btw',PQ(1),PQ(2),D0,2);
H = 0.5 + 2*HBW;
gbw = dftfilt(gf,HBW);
gbw = gscale(gbw);
ghf = dftfilt(gf,H);
ghf = gscale(ghf);
ghe = histeq(ghf,256);
figure('name','3.3-1')
imshow(ghf)
imwrite(ghf,['D:\图像处理\','3.3-1高频强调','.png'])

figure('name','3.3-2')
imshow(ghe)
imwrite(ghe,['D:\图像处理\','3.3-2高频强调+直方图均衡','.png'])

%4.1维纳滤波

%%4.1.1获得模糊退化且添加高斯噪声的图像
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
I = rgb2gray(f);
[m,n] = size(I);
F = fftshift(fft2(I));
k = 0.005;
for u=1:m
    for v=1:n
        H(u,v)=exp((-k)*(((u-m/2)^2+(v-n/2)^2)^(5/6)));
        
    end
end
G = F.*H;
I0 = real(ifft2(fftshift(G)));
I1 = imnoise(uint8(I0),'gaussian',0,0.001);
figure('name','4.1.1')
imshow(uint8(I1))
imwrite(uint8(I1),['D:\图像处理\','4.1.1模糊退化且添加高斯噪声的图像','.png'])
%title('模糊退化且添加高斯噪声的图像')

%%4.1.3进行维纳滤波
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
I = rgb2gray(f);
LEN = 21;
THETA = 11;
PSF = fspecial('motion',LEN,THETA);
blurred = imfilter(I,PSF,'conv','circular');
wnr = deconvwnr(blurred,PSF);
figure('name','4.1.3')
imshow(wnr)
imwrite(wnr,['D:\图像处理\','4.1.3维纳滤波','.png'])

%4.2仿射变换  theta为变换角度可调节
clear
f = imread("D:\matlab-m\原图.png");
gf = rgb2gray(f);
f = checkerboard(50);
s = 0.8;
theta  = pi/6;
T = [s*cos(theta) s*sin(theta) 0;-s*sin(theta) s*cos(theta) 0;0 0 1 ];
tform = maketform('affine',T);
g = imtransform(gf,tform);
figure('name','4.2-1')
imshow(g)
imwrite(g,['D:\图像处理\','4.2-1仿射变换为六分之Π','.png'])

theta  = pi/3;
T = [s*cos(theta) s*sin(theta) 0;-s*sin(theta) s*cos(theta) 0;0 0 1 ];
tform = maketform('affine',T);
g = imtransform(gf,tform);
figure('name','4.2-2')
imshow(g)
imwrite(g,['D:\图像处理\','4.2-2仿射变换为三分之Π','.png'])

theta  = pi/2;
T = [s*cos(theta) s*sin(theta) 0;-s*sin(theta) s*cos(theta) 0;0 0 1 ];
tform = maketform('affine',T);
g = imtransform(gf,tform);
figure('name','4.2-3')
imshow(g)
imwrite(g,['D:\图像处理\','4.2-3仿射变换为二分之Π','.png'])

%5.1颜色模型转换

%%5.1.1RGB模型
%8色索引为-1
%16色为-2
%32色为-3
%256色为-4

clear

RGB = imread("D:\matlab-m\原图.png");
[R_i,map] = rgb2ind(RGB,8);
figure('name','5.1.1-1')
imshow(R_i,map)
imwrite(R_i,map,['D:\图像处理\','5.1.1-1-8色索引','.png'])

[R_i,map] = rgb2ind(RGB,16);
figure('name','5.1.1-2')
imshow(R_i,map)
imwrite(R_i,map,['D:\图像处理\','5.1.1-2-16色索引','.png'])

[R_i,map] = rgb2ind(RGB,32);
figure('name','5.1.1-3')
imshow(R_i,map)
imwrite(R_i,map,['D:\图像处理\','5.1.1-3-32色索引','.png'])

[R_i,map] = rgb2ind(RGB,256);
figure('name','5.1.1-4')
imshow(R_i,map)
imwrite(R_i,map,['D:\图像处理\','5.1.1-4-256色索引','.png'])

%%5.1.2RGB分量
%R为-1
%G为-2
%B为-3
PR = RGB(:,:,1);
PG = RGB(:,:,2);
PB = RGB(:,:,3);
figure('name','5.1.2-1')
imshow(PR)
imwrite(PR,['D:\图像处理\','5.1.2-1RGB-R','.png'])

figure('name','5.1.2-2')
imshow(PG)
imwrite(PG,['D:\图像处理\','5.1.2-2RGB-G','.png'])

figure('name','5.1.2-3')
imshow(PB)
imwrite(PB,['D:\图像处理\','5.1.2-3RGB-B','.png'])

%%5.2.1RGB转化为HSV
%HSV为-1

clear
RGB  = imread("D:\matlab-m\原图.png");
RGB_hsv = rgb2hsv(RGB);
figure('name','5.2.1-1')
imshow(RGB_hsv)
imwrite(RGB_hsv,['D:\图像处理\','5.2.1-1HSV','.png'])

%%5.2.2HSV分量
%H为-1
%S为-2
%V为-3
H = RGB_hsv(:,:,1);
S = RGB_hsv(:,:,2);
V = RGB_hsv(:,:,3);
figure('name','5.2.2-1')
imshow(H)
imwrite(H,['D:\图像处理\','5.2.2-1HSV-H','.png'])

figure('name','5.2.2-2')
imshow(S)
imwrite(S,['D:\图像处理\','5.2.2-2HSV-S','.png'])

figure('name','5.2.2-3')
imshow(V)
imwrite(V,['D:\图像处理\','5.2.2-3HSV-V','.png'])


%5.3.1RGB转化为NTSC
%NTSC为-1
%Y分量为-2
%I为-3
%Q为-4
clear
RGB = imread("D:\matlab-m\原图.png");

RGB_ntsc = rgb2ntsc(RGB);
figure('name','5.3.1-1')
imshow(RGB_ntsc)
imwrite(RGB_ntsc,['D:\图像处理\','5.3.1-1NTSC','.png'])

Y = RGB_ntsc(:,:,1);
I = RGB_ntsc(:,:,2);
Q = RGB_ntsc(:,:,3);
figure('name','5.3.1-2')
imshow(Y)
imwrite(Y,['D:\图像处理\','5.3.1-2NTSC-Y','.png'])

figure('name','5.3.1-3')
imshow(I)
imwrite(I,['D:\图像处理\','5.3.1-3NTSC-I','.png'])

figure('name','5.3.1-4')
imshow(Q)
imwrite(Q,['D:\图像处理\','5.3.1-4NTSC-Q','.png'])

%%5.4.1RGB转化为YCbCr
%YCbCr为-1
%Y分量为-2
%Cb为-3
%Cr为-4
clear
RGB = imread("D:\matlab-m\原图.png");

RGB_ycbcr = rgb2ycbcr(RGB);
figure('name','5.4.1-1')
imshow(RGB_ycbcr)
imwrite(RGB_ycbcr,['D:\图像处理\','5.4.1-1YCbCr','.png'])

Y= RGB_ycbcr(:,:,1);
Cb = RGB_ycbcr(:,:,2);
Cr = RGB_ycbcr(:,:,3);
figure('name','5.4.1-2')
imshow(Y)
imwrite(Y,['D:\图像处理\','5.4.1-2YCbCr-Y','.png'])

figure('name','5.4.1-3')
imshow(Cb)
imwrite(Cb,['D:\图像处理\','5.4.1-3YCbCr-Cb','.png'])

figure('name','5.4.1-4')
imshow(Cr)
imwrite(Cr,['D:\图像处理\','5.4.1-4YCbCr-Cr','.png'])

%%5.5.1RGB转化为CMY
%CMY为-1
%C分量为-2
%M为-3
%Y为-4
clear
RGB = imread("D:\matlab-m\原图.png");

RGB_cmy = imcomplement(RGB);
figure('name','5.5.1-1')
imshow(RGB_cmy)
imwrite(RGB_cmy,['D:\图像处理\','5.5.1-1CMY','.png'])

C= RGB_cmy(:,:,1);
M = RGB_cmy(:,:,2);
Y = RGB_cmy(:,:,3);
figure('name','5.5.1-2')
imshow(C)
imwrite(C,['D:\图像处理\','5.5.1-2CMY-C','.png'])

figure('name','5.5.1-3')
imshow(M)
imwrite(M,['D:\图像处理\','5.5.1-3CMY-M','.png'])

figure('name','5.5.1-4')
imshow(Y)
imwrite(Y,['D:\图像处理\','5.5.1-4CMY-Y','.png'])

%%5.6.1RGB转化为HSI
%HSI为-1
%H分量为-2
%S为-3
%I为-4
clear
RGB = imread("D:\matlab-m\原图.png");

RGB_hsi = rgb2hsi(RGB);
figure('name','5.6.1-1')
imshow(RGB_hsi)
imwrite(RGB_hsi,['D:\图像处理\','5.6.1-1HSI','.png'])

H = RGB_hsi(:,:,1);
S = RGB_hsi(:,:,2);
I = RGB_hsi(:,:,3);
figure('name','5.6.1-2')
imshow(H)
imwrite(H,['D:\图像处理\','5.6.1-2HSI-H','.png'])

figure('name','5.6.1-3')
imshow(S)
imwrite(S,['D:\图像处理\','5.6.1-3HSI-S','.png'])

figure('name','5.6.1-4')
imshow(I)
imwrite(I,['D:\图像处理\','5.6.1-4HSI-I','.png'])



%
function [out,revertclass] = tofloat(inputimage)
%Copy the book of Gonzales
identify = @(x) x;
tosingle = @im2single;
table = {'uint8',tosingle,@im2uint8
'uint16',tosingle,@im2uint16
'logical',tosingle,@logical
'double',identify,identify
'single',identify,identify};
classIndex = find(strcmp(class(inputimage),table(:,1)));
if isempty(classIndex)
error('不支持的图像类型');
end
out = table{classIndex,2}(inputimage);
revertclass = table{classIndex,3};
end
%


%
function PQ = paddedsize(AB, CD, PARAM)
% 计算填充尺寸以供基于FFT的滤波器
% PQ = PADDEDSIZE(AB),AB = [A B], PQ = 2 * AB
%
% PQ = PADDEDSIZE(AB, 'PWR2')， PQ（1） = PQ（2） = 2 ^ nextpow2(2 * m), m =
% MAX(AB).
% 
% PQ = PADDEDSIZE(AB, CD)，AB = [A B], CD = [C D]
%
%  PQ = PADDEDSIZE(AB, CD, 'PWR2')， PQ（1） = PQ（2） = 2 ^ nextpow2(2 * m), m =
% MAX([AB CD]).

if nargin == 1
    PQ = 2 * AB;
elseif nargin == 2 & ~ischar(CD)
    PQ = AB + CD -1;
    PQ = 2 * ceil(PQ / 2);  % ceil（N）返回比N大的最小整数，为了避免出现奇数，因为处理偶数数组快
elseif nargin == 2
    m = max(AB);
    P = 2 ^ nextpow2(2 * m);  % nextpow2（N）返回第一个P，使得2. ^ P> = abs（N）。 
    % 对于FFT运算，找到最接近两个序列长度的2 的幂次方通常很有用。
    PQ = [P, P];
elseif nargin == 3
    m = max([AB CD]);
    P = 2 ^ nextpow2(2 * m);
    PQ = [P, P];
else
    error('Wrong number of input')
end
end
%


%
function [ U,V ] = dftuv( M, N )
%DFTUV 实现频域滤波器的网格函数
%   Detailed explanation goes here
u = 0:(M - 1);
v = 0:(N - 1);
idx = find(u > M/2); %找大于M/2的数据
u(idx) = u(idx) - M; %将大于M/2的数据减去M
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);      

end
%


%
function g = dftfilt(f,H)
% 此函数可接受输入图像和一个滤波函数，可处理所有的
% 滤波细节并输出经滤波和剪切后的图像
% 将此.m文件保存在一个文件夹
% file->set path->add with subfolder
% 将你函数所在文件夹添加到搜索路径
% save就可以将其添加到你的函数库了
F=fft2(f,size(H,1),size(H,2));
g=real(ifft2(H.*F));
g=g(1:size(f,1),1:size(f,2));
end
%


%
function H = hpfilter(type, M, N, D0, n)
% LPFILTER Computes frequency domain highpass filters
%   H = HPFILTER(TYPE, M, N, D0, n) creates the transfer function of a
%   highpass filter, H, of the specified TYPE and size (M-by-N). To view the
%   filter as an image or mesh plot, it should be centered using H =
%   fftshift(H)
%   Valid value for TYPE, D0, and n are:
%   'ideal' Ideal highpass filter with cutoff frequency D0. n need not be
%           supplied. D0 must be positive.
%   'btw'   Butterworth highpass filter of order n, and cutoff D0. The
%           default value for n is 1.0. D0 must be positive.
%   'gaussian'Gaussian highpass filter with cutoff (standard deviation) D0.
%           n need not be supplied. D0 must be positive.
% The transfer function Hhp of highpass filter is 1 - Hlp, where Hlp is the
% transfer function of the corresponding lowpass filter. Thus, we can use
% function lpfilter to generate highpass filters.
%
% 计算给定类型（理想、巴特沃兹、高斯）的频域高通滤波器

% Use function dftuv to set up the meshgrid arrays needed for computing the
% required distances.
if nargin == 4
    n = 1;
end
Hlp = lpfilter(type, M, N, D0, n);
H = 1 - Hlp;      
end
%


%
function [ H, D ] = lpfilter( type,M,N,D0,n )
%LPFILTER creates the transfer function of a lowpass filter.
%   Detailed explanation goes here

%use function dftuv to set up the meshgrid arrays needed for computing 
%the required distances.
[U, V] = dftuv(M,N);
 
%compute the distances D(U,V)
D = sqrt(U.^2 + V.^2);

%begin filter computations
switch type
    case 'ideal'
        H = double(D <= D0);
    case 'btw'
        if nargin == 4
            n = 1;
        end
        H = 1./(1+(D./D0).^(2*n));
    case 'gaussian'
        H = exp(-(D.^2)./(2*(D0^2)));
    otherwise 
        error('Unkown filter type');

end
end

%


%
function g=gscale(f,varargin)
if length(varargin)==0
  method='full8';
else method=varargin{1};
end
if strcmp(class(f),'double')&(max(f(:))>1 | min(f(:))<0)
   f=mat2gray(f);
end

switch method
case 'full8'
        g=im2uint8(mat2gray(double(f)));
case 'full16'
        g=im2uint16(mat2gray(double(f)));
case 'minmax'
       low = varargin{2};high = varargin{3};
       if low>1 | low<0 |high>1 | high<0
             error('Parameters low and high must be in the range [0,1]')
       end
       if strcmp(class(f),'double')
            low_in=min(f(:));
            high_in=max(f(:));
       elseif  strcmp(class(f),'uint8')
            low_in=double(min(f(:)))./255;
            high_in=double(max(f(:)))./255;
       elseif   strcmp(class(f),'uint16')
            low_in=double(min(f(:)))./65535;
            high_in=double(max(f(:)))./65535;
       end
       
       g=imadjust(f,[low_in high_in],[low high]);
otherwise
       error('Unknown method')
end
end
%

%
function hsi = rgb2hsi(rgb) 
%RGB2HSI Converts an RGB image to HSI. 
%   HSI = RGB2HSI(RGB) converts an RGB image to HSI. The input image 
%   is assumed to be of size M-by-N-by-3, where the third dimension 
%   accounts for three image planes: red, green, and blue, in that 
%   order. If all RGB component images are equal, the HSI conversion 
%   is undefined. The input image can be of class double (with values 
%   in the range [0, 1]), uint8, or uint16.  
% 
%   The output image, HSI, is of class double, where: 
%     hsi(:, :, 1) = hue image normalized to the range [0, 1] by 
%                    dividing all angle values by 2*pi.  
%     hsi(:, :, 2) = saturation image, in the range [0, 1]. 
%     hsi(:, :, 3) = intensity image, in the range [0, 1]. 

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins 
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004 
%   $Revision: 1.4 $  $Date: 2003/09/29 15:21:54 $ 

% Extract the individual component immages. 
rgb = im2double(rgb); 
r = rgb(:, :, 1); 
g = rgb(:, :, 2); 
b = rgb(:, :, 3); 

% Implement the conversion equations. 
num = 0.5*((r - g) + (r - b)); 
den = sqrt((r - g).^2 + (r - b).*(g - b)); 
theta = acos(num./(den + eps)); 

H = theta; 
H(b > g) = 2*pi - H(b > g); 
H = H/(2*pi); 

num = min(min(r, g), b); 
den = r + g + b; 
den(den == 0) = eps; 
S = 1 - 3.* num./den; 

H(S == 0) = 0; 

I = (r + g + b)/3; 

% Combine all three results into an hsi image. 
hsi = cat(3, H, S, I); 
end
%
