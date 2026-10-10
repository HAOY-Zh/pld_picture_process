function img_out = window_level(img_hu, WL, WW)
%WINDOW_LEVEL 对 HU 图像进行窗宽窗位处理，并映射到 0~1

low = WL - WW / 2;
high = WL + WW / 2;

img_out = double(img_hu);
img_out(img_out < low) = low;
img_out(img_out > high) = high;

img_out = (img_out - low) / (high - low);
img_out = double(img_out);
end
