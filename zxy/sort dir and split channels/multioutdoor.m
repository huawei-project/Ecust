clc;
clear;
%% 需要预设路径，需要修改
% 总共张35张(7*(4+1))
fileDir = 'D:/Desktop/42/'; %拍摄图片所在文件夹
outputDir = 'D:/Desktop/Outdoor/42/Multi/'; %图片保存文件夹
outputDirorign = 'D:/Desktop/Outdoor/orign/42/'; %更改输出目录，移动bsq，tif，hdr等文件

%% 图片分类
% 总共需要生成的目录
imgdir11 = [outputDir,'Multi_7_W1_1']; 
imgdir12 = [outputDir,'Multi_6_W1_1']; 
imgdir13 = [outputDir,'Multi_5_W1_1']; 
imgdir14 = [outputDir,'Multi_4_W1_1']; 
imgdir15 = [outputDir,'Multi_3_W1_1']; 
imgdir16 = [outputDir,'Multi_2_W1_1']; 
imgdir17 = [outputDir,'Multi_1_W1_1'];

imgdir21 = [outputDir,'Multi_7_W1_6']; 
imgdir22 = [outputDir,'Multi_6_W1_6']; 
imgdir23 = [outputDir,'Multi_5_W1_6']; 
imgdir24 = [outputDir,'Multi_4_W1_6']; 
imgdir25 = [outputDir,'Multi_3_W1_6']; 
imgdir26 = [outputDir,'Multi_2_W1_6']; 
imgdir27 = [outputDir,'Multi_1_W1_6']; 

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
for i =1:(length(ind))
    oldpath =[fileDir,a(i).name]; %输入图片
    name_number = rem(i,5); %第几张图片
    if name_number == 0 % 戴墨镜
        angle = ceil(i/5); %选择角度
        newdir = eval(['imgdir',mat2str(2),mat2str(angle)]); % 戴墨镜的目录
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
        
    elseif name_number ~= 0  % 不带墨镜
        angle = ceil(i/5); %选择角度
        imgdir = eval(['imgdir',mat2str(1),mat2str(angle)]); % 戴墨镜的目录
        newdir = [imgdir,'/',mat2str(name_number)];
       
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
end
%% 图片分类
% 总共需要生成的目录
imgdir31 = [outputDirorign,'Multi_1_W1_1']; 
imgdir32 = [outputDirorign,'Multi_2_W1_1']; 
imgdir33 = [outputDirorign,'Multi_3_W1_1']; 
imgdir34 = [outputDirorign,'Multi_4_W1_1']; 
imgdir35 = [outputDirorign,'Multi_5_W1_1']; 
imgdir36 = [outputDirorign,'Multi_6_W1_1']; 
imgdir37 = [outputDirorign,'Multi_7_W1_1'];

imgdir41 = [outputDirorign,'Multi_1_W1_6']; 
imgdir42 = [outputDirorign,'Multi_2_W1_6']; 
imgdir43 = [outputDirorign,'Multi_3_W1_6']; 
imgdir44 = [outputDirorign,'Multi_4_W1_6']; 
imgdir45 = [outputDirorign,'Multi_5_W1_6']; 
imgdir46 = [outputDirorign,'Multi_6_W1_6']; 
imgdir47 = [outputDirorign,'Multi_7_W1_6']; 

%% 从拍摄文件夹中按照拍摄时间读取所有bsq文件
filePattern = [fileDir, '*.bsq']; %图片格式
dirOutput = dir(filePattern); %获取图片名称字符串，并按时间排序
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind 图片数量排序
a = dirOutput(ind); 
% 图片分类

for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %输入图片
    name_number = rem(i,5); %第几张图片
    if name_number ~= 0  % 不带墨镜
        angle = ceil(i/5);
        imgdir = eval(['imgdir',mat2str(3),mat2str(angle)]);
        newdir = [imgdir,'/',mat2str(name_number)];
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
   
        newpath = [newdir,'/',a(i).name]; %之后的图片名称
        movefile(oldpath,newpath); %移到目标文件夹
        
    elseif name_number == 0

        angle = ceil(i/5) %选择角度
        imgdir = eval(['imgdir',mat2str(4),mat2str(angle)]);%之后的图片路径
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %之后的图片名称
        movefile(oldpath,newpath); %移到目标文件夹
    end
end
%% 从拍摄文件夹中按照拍摄时间读取所有hdr文件
filePattern = [fileDir, '*.hdr']; %图片格式
dirOutput = dir(filePattern); %获取图片名称字符串，并按时间排序
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind 图片数量排序
a = dirOutput(ind); 
% 图片分类

for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %输入图片
    name_number = rem(i,5); %第几张图片
    if name_number ~= 0  % 不带墨镜
        angle = ceil(i/5);
        imgdir = eval(['imgdir',mat2str(3),mat2str(angle)]);
        newdir = [imgdir,'/',mat2str(name_number)];
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
   
        newpath = [newdir,'/',a(i).name]; %之后的图片名称
        movefile(oldpath,newpath); %移到目标文件夹
        
    elseif name_number == 0

        angle = ceil(i/5) %选择角度
        imgdir = eval(['imgdir',mat2str(4),mat2str(angle)]);%之后的图片路径
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %之后的图片名称
        movefile(oldpath,newpath); %移到目标文件夹
    end
end
%% 从拍摄文件夹中按照拍摄时间读取所有tif文件
filePattern = [fileDir, '*.tif']; %图片格式
dirOutput = dir(filePattern); %获取图片名称字符串，并按时间排序
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind 图片数量排序
a = dirOutput(ind); 
% 图片分类

for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %输入图片
    name_number = rem(i,5); %第几张图片
    if name_number ~= 0  % 不带墨镜
        angle = ceil(i/5)
        imgdir = eval(['imgdir',mat2str(3),mat2str(angle)]);
        newdir = [imgdir,'/',mat2str(name_number)];
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
   
        newpath = [newdir,'/',a(i).name]; %之后的图片名称
        movefile(oldpath,newpath); %移到目标文件夹
        
    elseif name_number == 0

        angle = ceil(i/5); %选择角度
        imgdir = eval(['imgdir',mat2str(4),mat2str(angle)]);%之后的图片路径
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %之后的图片名称
        movefile(oldpath,newpath); %移到目标文件夹
    end
end