% TEST_PSEUDOCOLOR 对比自定义 8 段映射和 MATLAB turbo 映射。

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

% 使用项目中已有的 Mayo CT 测试文件。

filename = fullfile(project_dir, 'data', 'mayo', 'L067', ...
    'L067_QD_3_SHARP_1.CT.0003.0001.2016.01.21.18.12.01.921560.405032914.IMA');

% 读取 DICOM/IMA 并转换为 HU。

img_hu = read_ct(filename);

% 为伪彩色单独设置 HU 显示范围，不调用 window_level.m。

pseudo_low = -1000;
pseudo_high = 400;
img_gray = (double(img_hu) - pseudo_low) / ...
    (pseudo_high - pseudo_low);
img_gray = min(max(img_gray, 0), 1);

% 使用自定义的 8 段分段线性伪彩色映射。

rgb_out = pseudocolor_map(img_gray);

% 使用 MATLAB 内置 turbo 颜色表作为视觉对照。

rgb_turbo = ind2rgb(gray2ind(img_gray, 256), turbo(256));

% 检查输出尺寸、数值范围和有限性。

expected_size = [size(img_gray), 3];
assert(isequal(size(rgb_out), expected_size), ...
    '自定义伪彩色输出尺寸不是 MxNx3。');
assert(all(isfinite(rgb_out(:))), ...
    '自定义伪彩色输出包含 NaN 或 Inf。');
assert(min(rgb_out(:)) >= 0 && max(rgb_out(:)) <= 1, ...
    '自定义 RGB 值超出 0~1 范围。');
assert(isequal(size(rgb_turbo), expected_size), ...
    'MATLAB turbo 输出尺寸不是 MxNx3。');

fprintf('img_gray size: %d x %d\n', size(img_gray, 1), size(img_gray, 2));
fprintf('Pseudo HU range: [%g, %g]\n', pseudo_low, pseudo_high);
fprintf('rgb_out size: %d x %d x %d\n', size(rgb_out, 1), ...
    size(rgb_out, 2), size(rgb_out, 3));

% 使用人工灰度值检查颜色端点和区间边界。

test_values = [0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0];
test_rgb = pseudocolor_map(test_values);


fprintf('\n自定义映射的人工灰度测试：\n');
for k = 1:numel(test_values)
    color = squeeze(test_rgb(1, k, :));
    fprintf('gray=%5.3f -> R=%4.2f, G=%4.2f, B=%4.2f\n', ...
        test_values(k), color(1), color(2), color(3));
end

% 显示灰度、自定义伪彩色和 MATLAB turbo 结果。


figure('Name', 'CT Pseudocolor Comparison');

subplot(1, 3, 1);
imshow(img_gray, []);
title('Gray with Pseudo HU Range');

subplot(1, 3, 2);
imshow(rgb_out);
title('Custom 8-Segment Map');

subplot(1, 3, 3);
imshow(rgb_turbo);
title('MATLAB Turbo Map');
