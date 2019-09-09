clc;
clear;
%% ��ҪԤ��·������Ҫ�޸�
% �ܹ���156�ţ�5*7 *4��+16
fileDir = 'D:/Desktop/15/'; %����ͼƬ�����ļ���
outputDir = 'D:/Desktop/PROCESSING/15/Multi/'; %ͼƬ�����ļ���
outputDirorign = 'D:/Desktop/orign/6/'; %�������Ŀ¼���ƶ�bsq��tif��hdr���ļ�

%% ͼƬ����
% �ܹ���Ҫ���ɵ�Ŀ¼
imgdir11 = [outputDir,'normal/','Multi_1_W1_1']; 
imgdir12 = [outputDir,'normal/','Multi_2_W1_1']; 
imgdir13 = [outputDir,'normal/','Multi_3_W1_1']; 
imgdir14 = [outputDir,'normal/','Multi_4_W1_1']; 
imgdir15 = [outputDir,'normal/','Multi_5_W1_1']; 
imgdir16 = [outputDir,'normal/','Multi_6_W1_1']; 
imgdir17 = [outputDir,'normal/','Multi_7_W1_1']; 

imgdir110 = [outputDir,'normal/','Multi_1_W1_5']; 
imgdir120 = [outputDir,'normal/','Multi_2_W1_5']; 
imgdir130 = [outputDir,'normal/','Multi_3_W1_5']; 
imgdir140 = [outputDir,'normal/','Multi_4_W1_5']; 
imgdir150 = [outputDir,'normal/','Multi_5_W1_5']; 
imgdir160 = [outputDir,'normal/','Multi_6_W1_5']; 
imgdir170 = [outputDir,'normal/','Multi_7_W1_5']; 

imgdir21 = [outputDir,'illum1/','Multi_1_W1_1']; 
imgdir22 = [outputDir,'illum1/','Multi_2_W1_1'];
imgdir23 = [outputDir,'illum1/','Multi_3_W1_1'];
imgdir24 = [outputDir,'illum1/','Multi_4_W1_1'];
imgdir25 = [outputDir,'illum1/','Multi_5_W1_1'];
imgdir26 = [outputDir,'illum1/','Multi_6_W1_1'];
imgdir27 = [outputDir,'illum1/','Multi_7_W1_1'];

imgdir210 = [outputDir,'illum1/','Multi_1_W1_5']; 
imgdir220 = [outputDir,'illum1/','Multi_2_W1_5'];
imgdir230 = [outputDir,'illum1/','Multi_3_W1_5'];
imgdir240 = [outputDir,'illum1/','Multi_4_W1_5'];
imgdir250 = [outputDir,'illum1/','Multi_5_W1_5'];
imgdir260 = [outputDir,'illum1/','Multi_6_W1_5'];
imgdir270 = [outputDir,'illum1/','Multi_7_W1_5'];


imgdir31 = [outputDir,'illum2/','Multi_1_W1_1'];
imgdir32 = [outputDir,'illum2/','Multi_2_W1_1'];
imgdir33 = [outputDir,'illum2/','Multi_3_W1_1'];
imgdir34 = [outputDir,'illum2/','Multi_4_W1_1'];
imgdir35 = [outputDir,'illum2/','Multi_5_W1_1'];
imgdir36 = [outputDir,'illum2/','Multi_6_W1_1'];
imgdir37 = [outputDir,'illum2/','Multi_7_W1_1'];

imgdir310 = [outputDir,'illum2/','Multi_1_W1_5'];
imgdir320 = [outputDir,'illum2/','Multi_2_W1_5'];
imgdir330 = [outputDir,'illum2/','Multi_3_W1_5'];
imgdir340 = [outputDir,'illum2/','Multi_4_W1_5'];
imgdir350 = [outputDir,'illum2/','Multi_5_W1_5'];
imgdir360 = [outputDir,'illum2/','Multi_6_W1_5'];
imgdir370 = [outputDir,'illum2/','Multi_7_W1_5'];

imgdir41 = [outputDir,'illum3/','Multi_1_W1_1'];
imgdir42 = [outputDir,'illum3/','Multi_2_W1_1'];
imgdir43 = [outputDir,'illum3/','Multi_3_W1_1'];
imgdir44 = [outputDir,'illum3/','Multi_4_W1_1'];
imgdir45 = [outputDir,'illum3/','Multi_5_W1_1'];
imgdir46 = [outputDir,'illum3/','Multi_6_W1_1'];
imgdir47 = [outputDir,'illum3/','Multi_7_W1_1'];

imgdir410 = [outputDir,'illum3/','Multi_1_W1_5'];
imgdir420 = [outputDir,'illum3/','Multi_2_W1_5'];
imgdir430 = [outputDir,'illum3/','Multi_3_W1_5'];
imgdir440 = [outputDir,'illum3/','Multi_4_W1_5'];
imgdir450 = [outputDir,'illum3/','Multi_5_W1_5'];
imgdir460 = [outputDir,'illum3/','Multi_6_W1_5'];
imgdir470 = [outputDir,'illum3/','Multi_7_W1_5'];

imgdir51 = [outputDir,'normal/','Multi_4_W1_6'];
imgdir52 = [outputDir,'illum1/','Multi_4_W1_6'];
imgdir53 = [outputDir,'illum2/','Multi_4_W1_6'];
imgdir54 = [outputDir,'illum3/','Multi_4_W1_6'];

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
for i =1:(length(ind)-16)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    
    name_number = rem(i,5); %�ڼ���ͼƬ
    if name_number ~= 0
        
        light = ceil(i/35); %ѡ����Ź�
        angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
        
        newdir = [imgdir,'/',mat2str(name_number)]; %֮���ͼƬ·��
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
        
    elseif name_number == 0
        
        light = ceil(i/35); %ѡ����Ź�
        angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle),'0']);%֮���ͼƬ·��
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
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
            imgPath = [imgdir, '/',mat2str(j),'.jpg'];    % ��ϱ���·����ͼƬ����
            imwrite(img,imgPath);                 % A����������õ��Ĵ�����ͼƬ����
        end  
    end
end

% ��ī����������
for i =(length(ind)-15):length(ind)
    light = ceil(i/35);
    angle = ceil((i-(length(ind)-16))/4);

    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    name_number = rem((i-(length(ind)-16)),4); %�ڼ���ͼƬ
    if name_number == 0
        name_number = 4;
    end

    newdir = [imgdir,'/',mat2str(name_number)]; %֮���ͼƬ·��
    if ~exist(newdir,'dir')
        mkdir(newdir);     
    end
    
    oldpath =[fileDir,a(i).name]; %����ͼƬ
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

%% ͼƬ����
% �ܹ���Ҫ���ɵ�Ŀ¼
imgdir11 = [outputDirorign,'normal/','Multi_1_W1_1']; 
imgdir12 = [outputDirorign,'normal/','Multi_2_W1_1']; 
imgdir13 = [outputDirorign,'normal/','Multi_3_W1_1']; 
imgdir14 = [outputDirorign,'normal/','Multi_4_W1_1']; 
imgdir15 = [outputDirorign,'normal/','Multi_5_W1_1']; 
imgdir16 = [outputDirorign,'normal/','Multi_6_W1_1']; 
imgdir17 = [outputDirorign,'normal/','Multi_7_W1_1']; 

imgdir110 = [outputDirorign,'normal/','Multi_1_W1_5']; 
imgdir120 = [outputDirorign,'normal/','Multi_2_W1_5']; 
imgdir130 = [outputDirorign,'normal/','Multi_3_W1_5']; 
imgdir140 = [outputDirorign,'normal/','Multi_4_W1_5']; 
imgdir150 = [outputDirorign,'normal/','Multi_5_W1_5']; 
imgdir160 = [outputDirorign,'normal/','Multi_6_W1_5']; 
imgdir170 = [outputDirorign,'normal/','Multi_7_W1_5']; 

imgdir21 = [outputDirorign,'illum1/','Multi_1_W1_1']; 
imgdir22 = [outputDirorign,'illum1/','Multi_2_W1_1'];
imgdir23 = [outputDirorign,'illum1/','Multi_3_W1_1'];
imgdir24 = [outputDirorign,'illum1/','Multi_4_W1_1'];
imgdir25 = [outputDirorign,'illum1/','Multi_5_W1_1'];
imgdir26 = [outputDirorign,'illum1/','Multi_6_W1_1'];
imgdir27 = [outputDirorign,'illum1/','Multi_7_W1_1'];

imgdir210 = [outputDirorign,'illum1/','Multi_1_W1_5']; 
imgdir220 = [outputDirorign,'illum1/','Multi_2_W1_5'];
imgdir230 = [outputDirorign,'illum1/','Multi_3_W1_5'];
imgdir240 = [outputDirorign,'illum1/','Multi_4_W1_5'];
imgdir250 = [outputDirorign,'illum1/','Multi_5_W1_5'];
imgdir260 = [outputDirorign,'illum1/','Multi_6_W1_5'];
imgdir270 = [outputDirorign,'illum1/','Multi_7_W1_5'];


imgdir31 = [outputDirorign,'illum2/','Multi_1_W1_1'];
imgdir32 = [outputDirorign,'illum2/','Multi_2_W1_1'];
imgdir33 = [outputDirorign,'illum2/','Multi_3_W1_1'];
imgdir34 = [outputDirorign,'illum2/','Multi_4_W1_1'];
imgdir35 = [outputDirorign,'illum2/','Multi_5_W1_1'];
imgdir36 = [outputDirorign,'illum2/','Multi_6_W1_1'];
imgdir37 = [outputDirorign,'illum2/','Multi_7_W1_1'];

imgdir310 = [outputDirorign,'illum2/','Multi_1_W1_5'];
imgdir320 = [outputDirorign,'illum2/','Multi_2_W1_5'];
imgdir330 = [outputDirorign,'illum2/','Multi_3_W1_5'];
imgdir340 = [outputDirorign,'illum2/','Multi_4_W1_5'];
imgdir350 = [outputDirorign,'illum2/','Multi_5_W1_5'];
imgdir360 = [outputDirorign,'illum2/','Multi_6_W1_5'];
imgdir370 = [outputDirorign,'illum2/','Multi_7_W1_5'];

imgdir41 = [outputDirorign,'illum3/','Multi_1_W1_1'];
imgdir42 = [outputDirorign,'illum3/','Multi_2_W1_1'];
imgdir43 = [outputDirorign,'illum3/','Multi_3_W1_1'];
imgdir44 = [outputDirorign,'illum3/','Multi_4_W1_1'];
imgdir45 = [outputDirorign,'illum3/','Multi_5_W1_1'];
imgdir46 = [outputDirorign,'illum3/','Multi_6_W1_1'];
imgdir47 = [outputDirorign,'illum3/','Multi_7_W1_1'];

imgdir410 = [outputDirorign,'illum3/','Multi_1_W1_5'];
imgdir420 = [outputDirorign,'illum3/','Multi_2_W1_5'];
imgdir430 = [outputDirorign,'illum3/','Multi_3_W1_5'];
imgdir440 = [outputDirorign,'illum3/','Multi_4_W1_5'];
imgdir450 = [outputDirorign,'illum3/','Multi_5_W1_5'];
imgdir460 = [outputDirorign,'illum3/','Multi_6_W1_5'];
imgdir470 = [outputDirorign,'illum3/','Multi_7_W1_5'];

imgdir51 = [outputDirorign,'normal/','Multi_4_W1_6'];
imgdir52 = [outputDirorign,'illum1/','Multi_4_W1_6'];
imgdir53 = [outputDirorign,'illum2/','Multi_4_W1_6'];
imgdir54 = [outputDirorign,'illum3/','Multi_4_W1_6'];

%% �������ļ����а�������ʱ���ȡ����bsq�ļ�
filePattern = [fileDir, '*.bsq']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 
% ͼƬ����

for i =1:(length(ind)-16)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,5); %�ڼ���ͼƬ
    if name_number ~= 0
        
        light = ceil(i/35); %ѡ����Ź�
        angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
        
        newdir = [imgdir,'/',mat2str(name_number)]; %֮���ͼƬ·��
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
        
        newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
        
    elseif name_number == 0
        
        light = ceil(i/35); %ѡ����Ź�
        angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle),'0']);%֮���ͼƬ·��
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
    end
end

% ��ī����������
for i =(length(ind)-15):length(ind)
    light = ceil(i/35);
    angle = ceil((i-(length(ind)-16))/4);

    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    name_number = rem((i-(length(ind)-16)),4); %�ڼ���ͼƬ
    if name_number == 0
        name_number = 4;
    end

    newdir = [imgdir,'/',mat2str(name_number)]; %֮���ͼƬ·��
    if ~exist(newdir,'dir')
        mkdir(newdir);     
    end
    
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
    movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
end


%% �������ļ����а�������ʱ���ȡ����hdr�ļ�
filePattern = [fileDir, '*.hdr']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 
% ͼƬ����

for i =1:(length(ind)-16)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,5); %�ڼ���ͼƬ
    if name_number ~= 0
        
        light = ceil(i/35); %ѡ����Ź�
        angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
        
        newdir = [imgdir,'/',mat2str(name_number)]; %֮���ͼƬ·��
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
        
        newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
        
    elseif name_number == 0
        
        light = ceil(i/35); %ѡ����Ź�
        angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle),'0']);%֮���ͼƬ·��
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
    end
end

% ��ī����������
for i =(length(ind)-15):length(ind)
    light = ceil(i/35);
    angle = ceil((i-(length(ind)-16))/4);

    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    name_number = rem((i-(length(ind)-16)),4); %�ڼ���ͼƬ
    if name_number == 0
        name_number = 4;
    end

    newdir = [imgdir,'/',mat2str(name_number)]; %֮���ͼƬ·��
    if ~exist(newdir,'dir')
        mkdir(newdir);     
    end
    
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
    movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
end
%% �������ļ����а�������ʱ���ȡ����tif�ļ�
filePattern = [fileDir, '*.tif']; %ͼƬ��ʽ
dirOutput = dir(filePattern); %��ȡͼƬ�����ַ���������ʱ������
[~, ind] = sort([dirOutput(:).datenum], 'ascend'); %ind ͼƬ��������
a = dirOutput(ind); 
% ͼƬ����

for i =1:(length(ind)-16)
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    name_number = rem(i,5); %�ڼ���ͼƬ
    if name_number ~= 0
        
        light = ceil(i/35); %ѡ����Ź�
        angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
        
        newdir = [imgdir,'/',mat2str(name_number)]; %֮���ͼƬ·��
        if ~exist(newdir,'dir')
            mkdir(newdir);     
        end
        
        newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
        
    elseif name_number == 0
        
        light = ceil(i/35); %ѡ����Ź�
        angle = ceil(i/5)-7*(light-1); %ѡ��Ƕ�
        imgdir = eval(['imgdir',mat2str(light),mat2str(angle),'0']);%֮���ͼƬ·��
        
        if ~exist(imgdir,'dir')
            mkdir(imgdir);     
        end
        
        newpath = [imgdir,'/',a(i).name]; %֮���ͼƬ����
        movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
    end
end

% ��ī����������
for i =(length(ind)-15):length(ind)
    light = ceil(i/35);
    angle = ceil((i-(length(ind)-16))/4);

    imgdir = eval(['imgdir',mat2str(light),mat2str(angle)]);
    name_number = rem((i-(length(ind)-16)),4); %�ڼ���ͼƬ
    if name_number == 0
        name_number = 4;
    end

    newdir = [imgdir,'/',mat2str(name_number)]; %֮���ͼƬ·��
    if ~exist(newdir,'dir')
        mkdir(newdir);     
    end
    
    oldpath =[fileDir,a(i).name]; %����ͼƬ
    newpath = [newdir,'/',a(i).name]; %֮���ͼƬ����
    movefile(oldpath,newpath); %�Ƶ�Ŀ���ļ���
end