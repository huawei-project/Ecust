clc;
clear;
%% ��ҪԤ��·������Ҫ�޸�
% �ܹ�128�ţ�4*7 *4��+16
fileDir = 'D:/Desktop/69rgb/'; %����ͼƬ�����ļ���
outputDir = 'E:/Indoor/69/rgb/'; %ͼƬ�����ļ���

%% �������ļ����а�������ʱ���ȡͼƬ
filePattern = [fileDir, '*.jpg']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 

%% ͼƬ����
% �ܹ���Ҫ���ɵ�Ŀ¼
imgdir11 = [outputDir,'normal/','RGB_1_W1_1']; 
imgdir12 = [outputDir,'normal/','RGB_2_W1_1']; 
imgdir13 = [outputDir,'normal/','RGB_3_W1_1']; 
imgdir14 = [outputDir,'normal/','RGB_4_W1_1']; 
imgdir15 = [outputDir,'normal/','RGB_5_W1_1']; 
imgdir16 = [outputDir,'normal/','RGB_6_W1_1']; 
imgdir17 = [outputDir,'normal/','RGB_7_W1_1']; 

imgdir21 = [outputDir,'illum1/','RGB_1_W1_1']; 
imgdir22 = [outputDir,'illum1/','RGB_2_W1_1'];
imgdir23 = [outputDir,'illum1/','RGB_3_W1_1'];
imgdir24 = [outputDir,'illum1/','RGB_4_W1_1'];
imgdir25 = [outputDir,'illum1/','RGB_5_W1_1'];
imgdir26 = [outputDir,'illum1/','RGB_6_W1_1'];
imgdir27 = [outputDir,'illum1/','RGB_7_W1_1'];

imgdir31 = [outputDir,'illum2/','RGB_1_W1_1'];
imgdir32 = [outputDir,'illum2/','RGB_2_W1_1'];
imgdir33 = [outputDir,'illum2/','RGB_3_W1_1'];
imgdir34 = [outputDir,'illum2/','RGB_4_W1_1'];
imgdir35 = [outputDir,'illum2/','RGB_5_W1_1'];
imgdir36 = [outputDir,'illum2/','RGB_6_W1_1'];
imgdir37 = [outputDir,'illum2/','RGB_7_W1_1'];

imgdir41 = [outputDir,'illum3/','RGB_1_W1_1'];
imgdir42 = [outputDir,'illum3/','RGB_2_W1_1'];
imgdir43 = [outputDir,'illum3/','RGB_3_W1_1'];
imgdir44 = [outputDir,'illum3/','RGB_4_W1_1'];
imgdir45 = [outputDir,'illum3/','RGB_5_W1_1'];
imgdir46 = [outputDir,'illum3/','RGB_6_W1_1'];
imgdir47 = [outputDir,'illum3/','RGB_7_W1_1']; 

imgdir51 = [outputDir,'normal/','RGB_4_W1_6'];
imgdir52 = [outputDir,'illum1/','RGB_4_W1_6'];
imgdir53 = [outputDir,'illum2/','RGB_4_W1_6'];
imgdir54 = [outputDir,'illum3/','RGB_4_W1_6'];

% ͼƬ����
for i =1:length(ind)
    light = ceil(i/28); %ѡ����Ź�
    angle = ceil(i/4)-7*(light-1); %ѡ��Ƕ�
    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    if ~exist(imgdir,'dir')
        mkdir(imgdir);     
    end
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,4); %�ڼ���ͼƬ
    if name_number == 0
        name_number = 4;
    end
    newpath = [imgdir,'/',mat2str(name_number),'.jpg']; %֮���ͼƬ����
    movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
end