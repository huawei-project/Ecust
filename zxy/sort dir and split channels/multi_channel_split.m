clc;
clear;
%% 需要预设路径，需要修改
filename = 'D:/Desktop/42/1.bsq/'; %拍摄图片所在文件夹
outputdir = 'D:/Desktop/Outdoor/42/Multi/'; %图片保存文件夹
%% % bsq按波段分开存放成jpg
bands = 25;%波段数
samples = 409;%列数
lines = 215;%行数
columns = samples*lines;%像元个数
%读取高光谱数据二进制文件
precision = 'uint16';
fp = fopen(filename,'r');
    %读取头文件
    image = fread(fp,[columns,bands],precision);
    image  = image';
    %关闭文件
    fclose(fp);
    for j = 1:bands
        img = reshape(image(j,:),[samples,lines]);
        img = uint8(double(img)/65535*255);
        imgPath = [outputdir, '/',mat2str(j),'.jpg'];    % 组合保存路径和图片名称
        imwrite(img,imgPath);                 % A假设就是所得到的待保存图片矩阵
    end