function img_hu = read_ct(filename)
%READ_CT 读取 DICOM/IMA CT 图像并转换为 HU 值。

info = dicominfo(filename);
img = double(dicomread(info));

if isfield(info, 'RescaleSlope')
    slope = double(info.RescaleSlope);
else
    slope = 1;
end

if isfield(info, 'RescaleIntercept')
    intercept = double(info.RescaleIntercept);
else
    intercept = 0;
end

img_hu = img .* slope + intercept;
end
