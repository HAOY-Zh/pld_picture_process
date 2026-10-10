% COMPARE_WINDOW_LEVEL_RTL 比较 MATLAB 和 Verilog RTL 窗宽窗位结果

% 两个文本文件每行一个灰度值，恢复图像时还原为行优先顺序

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
mat_file = fullfile(script_dir, 'window_level_test_data.mat');
expected_file = fullfile(script_dir, 'matlab_expected.txt');
rtl_file = fullfile(script_dir, 'rtl_output.txt');

if ~isfile(mat_file)
    error('找不到测试数据，请先运行 export_window_level_test.m');
end
if ~isfile(expected_file)
    error('找不到 MATLAB 参考文件: %s', expected_file);
end
if ~isfile(rtl_file)
    error('找不到 RTL 输出文件，请先运行 Vivado/XSim testbench: %s', rtl_file);
end

data = load(mat_file, 'img_hu_patch');
matlab_expected = read_integer_lines(expected_file);
rtl_output = read_integer_lines(rtl_file);

expected_count = 32 * 32;
if numel(matlab_expected) ~= expected_count
    error('MATLAB 参考像素数错误：期望 %d，实际 %d', ...
        expected_count, numel(matlab_expected));
end
if numel(rtl_output) ~= expected_count
    error('RTL 输出像素数错误：期望 %d，实际 %d', ...
        expected_count, numel(rtl_output));
end

matlab_expected = double(matlab_expected(:));
rtl_output = double(rtl_output(:));
absolute_error = abs(matlab_expected - rtl_output);

same_count = sum(absolute_error == 0);
different_count = sum(absolute_error ~= 0);
max_error = max(absolute_error);
mean_error = mean(absolute_error);
within_one_count = sum(absolute_error <= 1);

fprintf('MATLAB 参考像素数: %d\n', numel(matlab_expected));
fprintf('RTL 输出像素数: %d\n', numel(rtl_output));
fprintf('完全相同像素数: %d\n', same_count);
fprintf('不同像素数: %d\n', different_count);
fprintf('最大绝对误差: %g\n', max_error);
fprintf('平均绝对误差: %.6f\n', mean_error);
fprintf('|误差| <= 1 像素数: %d\n', within_one_count);

% 输出前 20 个误差不为 0 的像素位置和数值

different_indices = find(absolute_error ~= 0);
display_count = min(numel(different_indices), 20);
if display_count > 0
    fprintf('\n前 %d 个不一致像素：\n', display_count);
    fprintf(' index row col MATLAB RTL error\n');
    for k = 1:display_count
        index = different_indices(k);
        row = floor((index - 1) / 32) + 1;
        col = mod(index - 1, 32) + 1;
        fprintf('%6d %3d %3d %7d %4d %5d\n', ...
            index, row, col, matlab_expected(index), ...
            rtl_output(index), rtl_output(index) - matlab_expected(index));
    end
end

% 将行优先向量恢复为 32x32 图像

img_rtl = transpose(reshape(rtl_output, 32, 32));
img_matlab = transpose(reshape(matlab_expected, 32, 32));
img_error = abs(img_matlab - img_rtl);

figure('Name', 'MATLAB vs RTL Window Level');

subplot(2, 2, 1);
imshow(data.img_hu_patch, []);
title('原始 HU 区域');
colormap gray;
colorbar;

subplot(2, 2, 2);
imshow(img_matlab, []);
title('MATLAB 窗宽窗位');

subplot(2, 2, 3);
imshow(img_rtl, []);
title('Verilog RTL 窗宽窗位');

subplot(2, 2, 4);
imshow(img_error, []);
title('绝对误差');
colorbar;

% 读取每行一个整数的文本文件

function values = read_integer_lines(filename_in)
    values = dlmread(filename_in);
    values = values(:);
end
