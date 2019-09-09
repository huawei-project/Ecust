clc;
clear;
%% ��ҪԤ��·������Ҫ�޸�
% �ܹ���35��(7*(4+1))
fileDir = 'D:/Desktop/42/'; %����ͼƬ�����ļ���
outputDir = 'D:/Desktop/Outdoor/42/Multi/'; %ͼƬ�����ļ���
outputDirorign = 'D:/Desktop/Outdoor/orign/42/'; %�������Ŀ¼���ƶ�bsq��tif��hdr���ļ�

%% ͼƬ����
% �ܹ���Ҫ���ɵ�Ŀ¼
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

%% % bsq�����ηֿ���ų�jpg
% �������ļ����а�������ʱ���ȡ����bsq�ļ�
filePattern = [fileDir, '*.bsq']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 

bands = 25;%������
samples = 409;%����
lines = 215;%����
columns = samples*lines;%��Ԫ����
%��ȡ�߹������ݶ������ļ�
precision = 'uint16';

% ͼƬ�ƶ�
for i =1:(length(ind))
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,5); %�ڼ���ͼƬ
    if name_number == 0 % ��ī��
        angle = ceil(i/5); %ѡ��Ƕ�
        newdir = eval(['imgdir',mat2str(2),mat2str(angle)]); % ��ī����Ŀ¼
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
        
        % �ֿ�����
        fp2 = fopen(oldpath,'r');
        %��ȡͷ�ļ�
        image = fread(fp2,[columns,bands],precision);
        image  =  image';
        %�ر��ļ�
        fclose(fp2);
        for j = 1:bands
            img = reshape(image(j,:),[samples,lines]);
            img = uint8(double(img)/65535*255);
            imgPath = [newdir, '/',mat2str(j),'.jpg'];    % ��ϱ���·����ͼƬ����
            imwrite(img,imgPath);                 % A����������õ��Ĵ�����ͼƬ����
        end
        
    elseif name_number ~= 0  % ����ī��
        angle = ceil(i/5); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(1),mat2str(angle)]); % ��ī����Ŀ¼
        newdir = [imgdir,'/',mat2str(name_number)];
       
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
        % �ֿ�����
        fp2 = fopen(oldpath,'r');
        %��ȡͷ�ļ�

        image = fread(fp2,[columns,bands],precision);
        image  =  image';
        %�ر��ļ�
        fclose(fp2);
        for j = 1:bands
            img = reshape(image(j,:),[samples,lines]);
            img = uint8(double(img)/65535*255);
            imgPath = [newdir, '/',mat2str(j),'.jpg'];    % ��ϱ���·����ͼƬ����
            imwrite(img,imgPath);                 % A����������õ��Ĵ�����ͼƬ����
        end  
    end
end
%% ͼƬ����
% �ܹ���Ҫ���ɵ�Ŀ¼
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

%% �������ļ����а�������ʱ���ȡ����bsq�ļ�
filePattern = [fileDir, '*.bsq']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 
% ͼƬ����

for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,5); %�ڼ���ͼƬ
    if name_number ~= 0  % ����ī��
        angle = ceil(i/5);
        imgdir = eval(['imgdir',mat2str(3),mat2str(angle)]);
        newdir = [imgdir,'/',mat2str(name_number)];
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
   
        newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
        
    elseif name_number == 0

        angle = ceil(i/5) %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(4),mat2str(angle)]);%֮���ͼƬ·��
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
    end
end
%% �������ļ����а�������ʱ���ȡ����hdr�ļ�
filePattern = [fileDir, '*.hdr']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 
% ͼƬ����

for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,5); %�ڼ���ͼƬ
    if name_number ~= 0  % ����ī��
        angle = ceil(i/5);
        imgdir = eval(['imgdir',mat2str(3),mat2str(angle)]);
        newdir = [imgdir,'/',mat2str(name_number)];
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
   
        newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
        
    elseif name_number == 0

        angle = ceil(i/5) %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(4),mat2str(angle)]);%֮���ͼƬ·��
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
    end
end
%% �������ļ����а�������ʱ���ȡ����tif�ļ�
filePattern = [fileDir, '*.tif']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 
% ͼƬ����

for i =1:length(ind)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,5); %�ڼ���ͼƬ
    if name_number ~= 0  % ����ī��
        angle = ceil(i/5)
        imgdir = eval(['imgdir',mat2str(3),mat2str(angle)]);
        newdir = [imgdir,'/',mat2str(name_number)];
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
   
        newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
        
    elseif name_number == 0

        angle = ceil(i/5); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(4),mat2str(angle)]);%֮���ͼƬ·��
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
    end
end