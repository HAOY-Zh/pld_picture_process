function rgb_out = pseudocolor_map(gray_img)
%PSEUDOCOLOR_MAP 使用 8 段分段线性映射生成 RGB 伪彩色图像。

% 输入为 0~1 的 double 灰度图，输出为 MxNx3 的 double RGB 图像。

gray_img = double(gray_img);
gray_img(isnan(gray_img)) = 0;
gray_img = min(max(gray_img, 0), 1);

[rows, cols] = size(gray_img);
rgb_out = zeros(rows, cols, 3);

% 9 个颜色控制点形成 8 个线性区间。
% 颜色从深蓝逐步经过蓝、青、绿、黄、橙，最后到红色。

control_points = [
    0.00, 0.00, 0.00, 0.50; ...
    0.125, 0.00, 0.25, 1.00; ...
    0.25, 0.00, 0.85, 1.00; ...
    0.375, 0.00, 1.00, 0.55; ...
    0.50, 0.00, 1.00, 0.00; ...
    0.625, 0.75, 1.00, 0.00; ...
    0.75, 1.00, 1.00, 0.00; ...
    0.875, 1.00, 0.35, 0.00; ...
    1.00, 0.75, 0.00, 0.00];

% 每个区间进行一次线性插值，便于后续转换为 FPGA 分段逻辑。

for segment = 1:size(control_points, 1) - 1
    x0 = control_points(segment, 1);
    x1 = control_points(segment + 1, 1);
    color0 = control_points(segment, 2:4);
    color1 = control_points(segment + 1, 2:4);

    if segment == size(control_points, 1) - 1
        mask = gray_img >= x0 & gray_img <= x1;
    else
        mask = gray_img >= x0 & gray_img < x1;
    end

    ratio = (gray_img - x0) / (x1 - x0);
    for channel = 1:3
        channel_image = rgb_out(:, :, channel);
        channel_image(mask) = color0(channel) + ...
            ratio(mask) * (color1(channel) - color0(channel));
        rgb_out(:, :, channel) = channel_image;
    end
end

rgb_out = min(max(rgb_out, 0), 1);
end
