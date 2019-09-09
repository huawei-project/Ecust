clc;
clear;
%% 需要预设路径，需要修改
% 总共张35张(7*(4+1))
fileDir = 'E:/Desktop/rgb/41/'; %拍摄图片所在文件夹
outputDir = 'D:/Desktop/Outdoor20190809/7/rgb/'; %图片保存文件夹

%% 从拍摄文件夹中按照拍摄时间读取图片
filePattern = [fileDir, '*.jpg']; %图片格式
dirOutput = dir(filePattern); %获取图片名称字符串，并按时间排序
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind 图片数量排序
a = dirOutput(ind); 

%% 图片分类
% 总共需要生成的目录
imgdir11 = [outputDir,'RGB_7_W1_1']; 
imgdir12 = [outputDir,'RGB_6_W1_1']; 
imgdir13 = [outputDir,'RGB_5_W1_1']; 
imgdir14 = [outputDir,'RGB_4_W1_1']; 
imgdir15 = [outputDir,'RGB_3_W1_1']; 
imgdir16 = [outputDir,'RGB_2_W1_1']; 
imgdir17 = [outputDir,'RGB_1_W1_1'];

imgdir21 = [outputDir,'RGB_7_W1_6']; 
imgdir22 = [outputDir,'RGB_6_W1_6']; 
imgdir23 = [outputDir,'RGB_5_W1_6']; 
imgdir24 = [outputDir,'RGB_4_W1_6']; 
imgdir25 = [outputDir,'RGB_3_W1_6']; 
imgdir26 = [outputDir,'RGB_2_W1_6']; 
imgdir27 = [outputDir,'RGB_1_W1_6']; 

%% 图片分类
for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %输入图片
    name_number = rem(i,5); %第几张图片
    angle = ceil(i/5); %选择角度
    
    if name_number ~= 0 % 没有戴墨镜
        imgdir = eval(['imgdir',mat2str(1),mat2str(angle)]);  
        if ~exist(imgdir,'dir')
            mkdir(imgdir);
        end
        newpath = [imgdir,'/',mat2str(name_number),'.jpg']; %之后的图片名称
        movefile(oldpath,newpath); %移到目标文件夹
        
    elseif name_number == 0 % 戴眼镜
        imgdir = eval(['imgdir',mat2str(2),mat2str(angle)]);%之后的图片路径
        newpath = [imgdir,'.jpg']; %之后的图片名称
        movefile(oldpath,newpath); %移到目标文件夹
    end  
end