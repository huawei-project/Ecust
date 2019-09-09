clc;
clear;
%% ��ҪԤ��·������Ҫ�޸�
% �ܹ���35��(7*(4+1))
fileDir = 'E:/Desktop/rgb/41/'; %����ͼƬ�����ļ���
outputDir = 'D:/Desktop/Outdoor20190809/7/rgb/'; %ͼƬ�����ļ���

%% �������ļ����а�������ʱ���ȡͼƬ
filePattern = [fileDir, '*.jpg']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 

%% ͼƬ����
% �ܹ���Ҫ���ɵ�Ŀ¼
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

%% ͼƬ����
for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,5); %�ڼ���ͼƬ
    angle = ceil(i/5); %ѡ��Ƕ�
    
    if name_number ~= 0 % û�д�ī��
        imgdir = eval(['imgdir',mat2str(1),mat2str(angle)]);  
        if ~exist(imgdir,'dir')
            mkdir(imgdir);
        end
        newpath = [imgdir,'/',mat2str(name_number),'.jpg']; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
        
    elseif name_number == 0 % ���۾�
        imgdir = eval(['imgdir',mat2str(2),mat2str(angle)]);%֮���ͼƬ·��
        newpath = [imgdir,'.jpg']; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
    end  
end