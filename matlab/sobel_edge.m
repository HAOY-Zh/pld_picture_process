function img_out = sobel_edge(img)
%SOBEL_EDGE 使用 Sobel 算子检测图像边缘。

img_out = edge(img, 'sobel');
end
