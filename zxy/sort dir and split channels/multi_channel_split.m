clc;
clear;
%% ��ҪԤ��·������Ҫ�޸�
filename = 'D:/Desktop/42/1.bsq/'; %����ͼƬ�����ļ���
outputdir = 'D:/Desktop/Outdoor/42/Multi/'; %ͼƬ�����ļ���
%% % bsq�����ηֿ���ų�jpg
bands = 25;%������
samples = 409;%����
lines = 215;%����
columns = samples*lines;%��Ԫ����
%��ȡ�߹������ݶ������ļ�
precision = 'uint16';
fp = fopen(filename,'r');
    %��ȡͷ�ļ�
    image = fread(fp,[columns,bands],precision);
    image  = image';
    %�ر��ļ�
    fclose(fp);
    for j = 1:bands
        img = reshape(image(j,:),[samples,lines]);
        img = uint8(double(img)/65535*255);
        imgPath = [outputdir, '/',mat2str(j),'.jpg'];    % ��ϱ���·����ͼƬ����
        imwrite(img,imgPath);                 % A����������õ��Ĵ�����ͼƬ����
    end