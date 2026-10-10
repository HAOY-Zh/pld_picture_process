% EXPORT_WINDOW_LEVEL_TEST 导出真实 CT 的 32x32 HU 测试数据

% 数据按行优先展开：先写第 1 行，再写第 2 行，直到第 32 行。

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);

% Mayo CT 测试图像路径

filename = fullfile(project_dir, 'data', 'mayo', 'L067', ...
    'L067_QD_3_SHARP_1.CT.0003.0001.2016.01.21.18.12.01.921560.405032914.IMA');

% 窗宽窗位测试参数

WL = 40;
WW = 400;

% 读取 DICOM/IMA 并转换为 HU

img_hu = read_ct(filename);

% 从图像中心截取 32x32 区域

[img_rows, img_cols] = size(img_hu);
patch_rows = 32;
patch_cols = 32;
row_start = floor((img_rows - patch_rows) / 2) + 1;
col_start = floor((img_cols - patch_cols) / 2) + 1;
img_hu_patch = img_hu(row_start:row_start + patch_rows - 1, ...
                      col_start:col_start + patch_cols - 1);

% 导出 HU 要求为整数，参考结果也基于同一份整数 HU 数据计算

img_hu_patch = round(img_hu_patch);

% 调用已有的 MATLAB 窗宽窗位函数

img_window = window_level(img_hu_patch, WL, WW);

% MATLAB 使用四舍五入转换 uint8，RTL 使用整数除法向下取整

% 因此两者最多可能相差 1 个灰度级

matlab_gray = uint8(round(img_window * 255));

% 转置后再按列展开，等价于原图像的行优先展开

hu_row_major = reshape(img_hu_patch.', [], 1);
expected_row_major = reshape(matlab_gray.', [], 1);

hu_file = fullfile(script_dir, 'hu_input.txt');
expected_file = fullfile(script_dir, 'matlab_expected.txt');

write_integer_lines(hu_file, hu_row_major);
write_integer_lines(expected_file, expected_row_major);

% 保存测试区域、参考结果和测试参数

mat_file = fullfile(script_dir, 'window_level_test_data.mat');
save(mat_file, 'img_hu_patch', 'img_window', 'matlab_gray', ...
    'hu_row_major', 'expected_row_major', 'WL', 'WW', ...
    'row_start', 'col_start', 'filename');

fprintf('HU 文件: %s\n', hu_file);
fprintf('MATLAB 参考文件: %s\n', expected_file);
fprintf('MAT 数据文件: %s\n', mat_file);
fprintf('导出像素数量: %d\n', numel(hu_row_major));

% 将整数向量逐行写入文本文件

function write_integer_lines(filename_out, values)
    fid = fopen(filename_out, 'w');
    if fid == -1
        error('无法打开输出文件: %s', filename_out);
    end

    cleanup_obj = onCleanup(@() fclose(fid));
    fprintf(fid, '%d\n', values);
end
