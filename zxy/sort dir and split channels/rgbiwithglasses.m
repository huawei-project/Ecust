clc;
clear;
%% ��ҪԤ��·������Ҫ�޸�
% �ܹ���156�ţ�5*7 *4��+16
fileDir = 'D:/Desktop/14/'; %����ͼƬ�����ļ���
outputDir = 'D:/Desktop/PROCESSING/14/RGB/'; %ͼƬ�����ļ���

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

imgdir110 = [outputDir,'normal/','RGB_1_W1_5']; 
imgdir120 = [outputDir,'normal/','RGB_2_W1_5']; 
imgdir130 = [outputDir,'normal/','RGB_3_W1_5']; 
imgdir140 = [outputDir,'normal/','RGB_4_W1_5']; 
imgdir150 = [outputDir,'normal/','RGB_5_W1_5']; 
imgdir160 = [outputDir,'normal/','RGB_6_W1_5']; 
imgdir170 = [outputDir,'normal/','RGB_7_W1_5']; 

imgdir21 = [outputDir,'illum1/','RGB_1_W1_1']; 
imgdir22 = [outputDir,'illum1/','RGB_2_W1_1'];
imgdir23 = [outputDir,'illum1/','RGB_3_W1_1'];
imgdir24 = [outputDir,'illum1/','RGB_4_W1_1'];
imgdir25 = [outputDir,'illum1/','RGB_5_W1_1'];
imgdir26 = [outputDir,'illum1/','RGB_6_W1_1'];
imgdir27 = [outputDir,'illum1/','RGB_7_W1_1'];

imgdir210 = [outputDir,'illum1/','RGB_1_W1_5']; 
imgdir220 = [outputDir,'illum1/','RGB_2_W1_5'];
imgdir230 = [outputDir,'illum1/','RGB_3_W1_5'];
imgdir240 = [outputDir,'illum1/','RGB_4_W1_5'];
imgdir250 = [outputDir,'illum1/','RGB_5_W1_5'];
imgdir260 = [outputDir,'illum1/','RGB_6_W1_5'];
imgdir270 = [outputDir,'illum1/','RGB_7_W1_5'];


imgdir31 = [outputDir,'illum2/','RGB_1_W1_1'];
imgdir32 = [outputDir,'illum2/','RGB_2_W1_1'];
imgdir33 = [outputDir,'illum2/','RGB_3_W1_1'];
imgdir34 = [outputDir,'illum2/','RGB_4_W1_1'];
imgdir35 = [outputDir,'illum2/','RGB_5_W1_1'];
imgdir36 = [outputDir,'illum2/','RGB_6_W1_1'];
imgdir37 = [outputDir,'illum2/','RGB_7_W1_1'];

imgdir310 = [outputDir,'illum2/','RGB_1_W1_5'];
imgdir320 = [outputDir,'illum2/','RGB_2_W1_5'];
imgdir330 = [outputDir,'illum2/','RGB_3_W1_5'];
imgdir340 = [outputDir,'illum2/','RGB_4_W1_5'];
imgdir350 = [outputDir,'illum2/','RGB_5_W1_5'];
imgdir360 = [outputDir,'illum2/','RGB_6_W1_5'];
imgdir370 = [outputDir,'illum2/','RGB_7_W1_5'];

imgdir41 = [outputDir,'illum3/','RGB_1_W1_1'];
imgdir42 = [outputDir,'illum3/','RGB_2_W1_1'];
imgdir43 = [outputDir,'illum3/','RGB_3_W1_1'];
imgdir44 = [outputDir,'illum3/','RGB_4_W1_1'];
imgdir45 = [outputDir,'illum3/','RGB_5_W1_1'];
imgdir46 = [outputDir,'illum3/','RGB_6_W1_1'];
imgdir47 = [outputDir,'illum3/','RGB_7_W1_1']; 

imgdir410 = [outputDir,'illum3/','RGB_1_W1_5'];
imgdir420 = [outputDir,'illum3/','RGB_2_W1_5'];
imgdir430 = [outputDir,'illum3/','RGB_3_W1_5'];
imgdir440 = [outputDir,'illum3/','RGB_4_W1_5'];
imgdir450 = [outputDir,'illum3/','RGB_5_W1_5'];
imgdir460 = [outputDir,'illum3/','RGB_6_W1_5'];
imgdir470 = [outputDir,'illum3/','RGB_7_W1_5'];

imgdir51 = [outputDir,'normal/','RGB_4_W1_6'];
imgdir52 = [outputDir,'illum1/','RGB_4_W1_6'];
imgdir53 = [outputDir,'illum2/','RGB_4_W1_6'];
imgdir54 = [outputDir,'illum3/','RGB_4_W1_6'];


%% ͼƬ����
for i =1:(length(ind)-16)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    
    name_number = rem(i,5); %�ڼ���ͼƬ
    light = ceil(i/35); %ѡ����Ź�
    angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
    
    if name_number ~= 0 % û�д��۾�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);  
        if ~exist(imgdir,'dir')
            mkdir(imgdir);
        end
        newpath = [imgdir,'/',mat2str(name_number),'.jpg']; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
        
    elseif name_number == 0 % ���۾�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle),'0']);%֮���ͼƬ·��
        newpath = [imgdir,'.jpg']; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
    end  
end

% ��ī����������
for i =(length(ind)-15):length(ind)
    light = ceil(i/35);
    angle = ceil((i-(length(ind)-16))/4);
    
    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    if ~exist(imgdir,'dir')
        mkdir(imgdir);     
    end
    name_number = rem((i-(length(ind)-16)),4); %�ڼ���ͼƬ
    if name_number == 0
        name_number = 4;
    end
    newpath = [imgdir,'/',mat2str(name_number),'.jpg']; %֮���ͼƬ·��
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
end