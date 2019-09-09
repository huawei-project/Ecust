clc;
clear;
%% 需要预设路径，需要修改
% 总共128张（4*7*4）+16
fileDir = 'D:/Desktop/6/'; %拍摄图片所在文件夹
outputDir = 'D:/Desktop/PROCESSING/6/Multi/'; %图片保存文件夹
outputDirorign = 'D:/Desktop/orign/6/'; %更改输出目录，移动bsq，tif，hdr等文件

%% 图片分类
% 总共需要生成的目录
imgdir11 = [outputDir,'normal/','Multi_1_W1_1']; 
imgdir12 = [outputDir,'normal/','Multi_2_W1_1']; 
imgdir13 = [outputDir,'normal/','Multi_3_W1_1']; 
imgdir14 = [outputDir,'normal/','Multi_4_W1_1']; 
imgdir15 = [outputDir,'normal/','Multi_5_W1_1']; 
imgdir16 = [outputDir,'normal/','Multi_6_W1_1']; 
imgdir17 = [outputDir,'normal/','Multi_7_W1_1']; 

imgdir21 = [outputDir,'illum1/','Multi_1_W1_1']; 
imgdir22 = [outputDir,'illum1/','Multi_2_W1_1'];
imgdir23 = [outputDir,'illum1/','Multi_3_W1_1'];
imgdir24 = [outputDir,'illum1/','Multi_4_W1_1'];
imgdir25 = [outputDir,'illum1/','Multi_5_W1_1'];
imgdir26 = [outputDir,'illum1/','Multi_6_W1_1'];
imgdir27 = [outputDir,'illum1/','Multi_7_W1_1'];

imgdir31 = [outputDir,'illum2/','Multi_1_W1_1'];
imgdir32 = [outputDir,'illum2/','Multi_2_W1_1'];
imgdir33 = [outputDir,'illum2/','Multi_3_W1_1'];
imgdir34 = [outputDir,'illum2/','Multi_4_W1_1'];
imgdir35 = [outputDir,'illum2/','Multi_5_W1_1'];
imgdir36 = [outputDir,'illum2/','Multi_6_W1_1'];
imgdir37 = [outputDir,'illum2/','Multi_7_W1_1'];

imgdir41 = [outputDir,'illum3/','Multi_1_W1_1'];
imgdir42 = [outputDir,'illum3/','Multi_2_W1_1'];
imgdir43 = [outputDir,'illum3/','Multi_3_W1_1'];
imgdir44 = [outputDir,'illum3/','Multi_4_W1_1'];
imgdir45 = [outputDir,'illum3/','Multi_5_W1_1'];
imgdir46 = [outputDir,'illum3/','Multi_6_W1_1'];
imgdir47 = [outputDir,'illum3/','Multi_7_W1_1']; 

imgdir51 = [outputDir,'normal/','Multi_4_W1_6'];
imgdir52 = [outputDir,'illum1/','Multi_4_W1_6'];
imgdir53 = [outputDir,'illum2/','Multi_4_W1_6'];
imgdir54 = [outputDir,'illum3/','Multi_4_W1_6'];

%% % bsq按波段分开存放成jpg
% 从拍摄文件夹中按照拍摄时间读取所有bsq文件
filePattern = [fileDir, '*.bsq']; %图片格式
dirOutput = dir(filePattern); %获取图片名称字符串，并按时间排序
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind 图片数量排序
a = dirOutput(ind); 

bands = 25;%波段数
samples = 409;%列数
lines = 215;%行数
columns = samples*lines;%像元个数
%读取高光谱数据二进制文件
precision = 'uint16';

% 图片移动
for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %输入图片
    
    light = ceil(i/28); %选择干扰光
    angle = ceil(i/4)-7*(light-1); %选择角度
    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    
    name_number = rem(i,4); %第几张图片
    if name_number == 0
        name_number = 4;
    end
    
    newdir = [imgdir,'/',mat2str(name_number)]; %之后的图片路径
    if ~exist(newdir,'dir')
        mkdir(newdir);     
    end
    
    % 分开过程
    fp2 = fopen(oldpath,'r');
    %读取头文件
    image = fread(fp2,[columns,bands],precision);
    image  =  image';
    %关闭文件
    fclose(fp2);
    for j = 1:bands
        img = reshape(image(j,:),[samples,lines]);
        img = uint8(double(img)/65535*255);
        imgPath = [newdir, '/',mat2str(j),'.jpg'];    % 组合保存路径和图片名称
        imwrite(img,imgPath);                 % A假设就是所得到的待保存图片矩阵
    end
end

%% 图片分类
% 总共需要生成的目录
imgdir11 = [outputDirorign,'normal/','Multi_1_W1_1']; 
imgdir12 = [outputDirorign,'normal/','Multi_2_W1_1']; 
imgdir13 = [outputDirorign,'normal/','Multi_3_W1_1']; 
imgdir14 = [outputDirorign,'normal/','Multi_4_W1_1']; 
imgdir15 = [outputDirorign,'normal/','Multi_5_W1_1']; 
imgdir16 = [outputDirorign,'normal/','Multi_6_W1_1']; 
imgdir17 = [outputDirorign,'normal/','Multi_7_W1_1']; 

imgdir21 = [outputDirorign,'illum1/','Multi_1_W1_1']; 
imgdir22 = [outputDirorign,'illum1/','Multi_2_W1_1'];
imgdir23 = [outputDirorign,'illum1/','Multi_3_W1_1'];
imgdir24 = [outputDirorign,'illum1/','Multi_4_W1_1'];
imgdir25 = [outputDirorign,'illum1/','Multi_5_W1_1'];
imgdir26 = [outputDirorign,'illum1/','Multi_6_W1_1'];
imgdir27 = [outputDirorign,'illum1/','Multi_7_W1_1'];

imgdir31 = [outputDirorign,'illum2/','Multi_1_W1_1'];
imgdir32 = [outputDirorign,'illum2/','Multi_2_W1_1'];
imgdir33 = [outputDirorign,'illum2/','Multi_3_W1_1'];
imgdir34 = [outputDirorign,'illum2/','Multi_4_W1_1'];
imgdir35 = [outputDirorign,'illum2/','Multi_5_W1_1'];
imgdir36 = [outputDirorign,'illum2/','Multi_6_W1_1'];
imgdir37 = [outputDirorign,'illum2/','Multi_7_W1_1'];

imgdir41 = [outputDirorign,'illum3/','Multi_1_W1_1'];
imgdir42 = [outputDirorign,'illum3/','Multi_2_W1_1'];
imgdir43 = [outputDirorign,'illum3/','Multi_3_W1_1'];
imgdir44 = [outputDirorign,'illum3/','Multi_4_W1_1'];
imgdir45 = [outputDirorign,'illum3/','Multi_5_W1_1'];
imgdir46 = [outputDirorign,'illum3/','Multi_6_W1_1'];
imgdir47 = [outputDirorign,'illum3/','Multi_7_W1_1']; 

imgdir51 = [outputDirorign,'normal/','Multi_4_W1_6'];
imgdir52 = [outputDirorign,'illum1/','Multi_4_W1_6'];
imgdir53 = [outputDirorign,'illum2/','Multi_4_W1_6'];
imgdir54 = [outputDirorign,'illum3/','Multi_4_W1_6'];

%% 从拍摄文件夹中按照拍摄时间读取所有bsq文件
filePattern = [fileDir, '*.bsq']; %图片格式
dirOutput = dir(filePattern); %获取图片名称字符串，并按时间排序
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind 图片数量排序
a = dirOutput(ind); 
% 图片分类
for i =1:length(ind)
    light = ceil(i/28); %选择干扰光
    angle = ceil(i/4)-7*(light-1); %选择角度 
    name_number = rem(i,4); %第几张图片
    if name_number == 0
        name_number = 4;
    end
    
    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    newdir = [imgdir,'/',mat2str(name_number)];
    if ~exist(newdir,'dir')
        mkdir(newdir);     
    end
    
    name_number = rem(i,4); %第几张图片
    if name_number == 0
        name_number = 4;
    end
    oldpath =[fileDir,a(i).name]; %输入图片
    newpath = [newdir,'/',a(i).name]; %之后的图片名称
    movefile(oldpath,newpath); %移到目标文件夹
end

%% 从拍摄文件夹中按照拍摄时间读取所有tif文件
filePattern = [fileDir, '*.tif']; %图片格式
dirOutput = dir(filePattern); %获取图片名称字符串，并按时间排序
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind 图片数量排序
a = dirOutput(ind); 
% 图片分类
for i =1:length(ind)
    light = ceil(i/28); %选择干扰光
    angle = ceil(i/4)-7*(light-1); %选择角度 
    name_number = rem(i,4); %第几张图片
    if name_number == 0
        name_number = 4;
    end
    
    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    newdir = [imgdir,'/',mat2str(name_number)];
    if ~exist(newdir,'dir')
        mkdir(newdir);     
    end
    
    name_number = rem(i,4); %第几张图片
    if name_number == 0
        name_number = 4;
    end
    oldpath =[fileDir,a(i).name]; %输入图片
    newpath = [newdir,'/',a(i).name]; %之后的图片名称
    movefile(oldpath,newpath); %移到目标文件夹
end
%% 从拍摄文件夹中按照拍摄时间读取所有hdr文件
filePattern = [fileDir, '*.hdr']; %图片格式
dirOutput = dir(filePattern); %获取图片名称字符串，并按时间排序
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind 图片数量排序
a = dirOutput(ind); 
% 图片分类
for i =1:length(ind)
    light = ceil(i/28); %选择干扰光
    angle = ceil(i/4)-7*(light-1); %选择角度 
    name_number = rem(i,4); %第几张图片
    if name_number == 0
        name_number = 4;
    end
    
    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    newdir = [imgdir,'/',mat2str(name_number)];
    if ~exist(newdir,'dir')
        mkdir(newdir);     
    end
    
    name_number = rem(i,4); %第几张图片
    if name_number == 0
        name_number = 4;
    end
    oldpath =[fileDir,a(i).name]; %输入图片
    newpath = [newdir,'/',a(i).name]; %之后的图片名称
    movefile(oldpath,newpath); %移到目标文件夹
end