function img_out = mean_filter(img)
%MEAN_FILTER 使用 3x3 均值滤波器处理图像。

h = fspecial('average', [3 3]);
img_out = imfilter(img, h, 'replicate');
end
