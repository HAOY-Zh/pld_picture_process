function img_out = median_filter(img)
%MEDIAN_FILTER 使用 3x3 中值滤波器处理图像。

img_out = medfilt2(img, [3 3]);
end
