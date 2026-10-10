% MAIN_BASIC_PROCESS 当前阶段仅测试 CT 窗宽窗位处理

% 请根据实际数据修改 CT 文件路径

filename = 'D:\pld_picture_process\data\mayo\L067\L067_QD_3_SHARP_1.CT.0003.0001.2016.01.21.18.12.01.921560.405032914.IMA';

% 第一组测试参数：腹部 CT 常用初始窗宽窗位

%WL = 40;
%WW = 400;

% 第二组测试参数，可手动取消注释进行测试

WL = -600;
WW = 1500;

% 读取 CT 图像并转换为 HU

img_hu = read_ct(filename);
fprintf('img_hu 范围: [%.2f, %.2f] HU\n', min(img_hu(:)), max(img_hu(:)));

% 进行窗宽窗位处理

img_window = window_level(img_hu, WL, WW);
fprintf('img_window 范围: [%.2f, %.2f]\n', min(img_window(:)), max(img_window(:)));

% 显示原始 HU 图像和窗宽窗位处理结果

figure('Name', 'CT Window Level Processing');

subplot(1, 2, 1);
imshow(img_hu, []);
colormap gray;
colorbar;
title('原始 HU 图像');

subplot(1, 2, 2);
imshow(img_window);
title(sprintf('窗宽窗位结果: WL=%g, WW=%g', WL, WW));
