function [images, labels] = load_MNIST_data(images_file, labels_file,...
    preprocess, is_show_images, varargin)
%����MNIST���ݼ���images��labels
% by ֣��ΰ Ewing 2016-04

if exist('is_show_images', 'var')
    images = load_MNIST_images(images_file, preprocess, is_show_images);
else
    images = load_MNIST_images(images_file);
end
labels = load_MNIST_labels(labels_file);

end

function images = load_MNIST_images(file_name, preprocess, is_show_images, varargin)
%����һ��  #���ص��� * #������ �ľ���

    %% ��ȡ raw MNIST images
    fp = fopen(file_name, 'rb');
    assert(fp ~= -1, ['Could not open ', file_name, ' ']);  % �򲻿��򱨴�

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', file_name, ' ']); % �涨�� magic number������check�ļ��Ƿ���ȷ

    num_images = fread(fp, 1, 'int32', 0, 'ieee-be'); % �����������������ļ��������Ե���
    num_rows   = fread(fp, 1, 'int32', 0, 'ieee-be');
    num_cols   = fread(fp, 1, 'int32', 0, 'ieee-be');

    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, num_cols, num_rows, num_images); % �ļ������ǰ������еģ���matlab�ǰ������еġ�
    images = permute(images, [ 2 1 3 ]);

    fclose(fp);
    %% ��ʾ200��images
    if exist('is_show_images', 'var') &&  is_show_images == 1
        figure('NumberTitle', 'off', 'Name', 'MNIST��д����ͼƬ');
        show_images_num = 200;
        penal           = show_images_num * 2 / 3;
        pic_Mat_Col     = ceil(1.5 * sqrt(penal));
        pic_Mat_Row     = ceil(show_images_num / pic_Mat_Col);
        for i = 1:show_images_num
            subplot(pic_Mat_Row, pic_Mat_Col, i, 'align');
            imshow(images(:, :, i));
        end
    end

    %% �� images ���д���
    % ת��Ϊ #���ص��� * #������ ����
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    
    if strcmp(preprocess, 'min_max_scaler')
        % ��һ���� [0,1]
        images = double(images) / 255; % �����ֵ��Ǹ�
    elseif strcmp(preprocess, 'zScore')
        % ��׼������
        images = zScore( images );% �����ֵ������ɸ�
    elseif strcmp(preprocess, 'whitening')
        % �׻�
        images = whitening( images ); % �����ֵ������ɸ�
    end
end

function data = zScore(data)
%�����ݽ��б�׼�����������������У�
% ȥ��ֵ��Ȼ�󷽲�����
    epsilon = 1e-8; % ��ֹ��0
    data = bsxfun(@minus, data, mean(data, 1)); % ȥ��ֵ����������ȥ��ͼƬ���ȣ�
    data = bsxfun(@rdivide, data, sqrt(mean(data .^ 2, 1)) + epsilon); % ȥ����
end
function data = whitening(data)
%�����ݽ��а׻����������������У�
% ȥ��ֵ��Ȼ��ȥ�����
    data = bsxfun(@minus, data, mean(data, 1)); % ȥ��ֵ
    [u, s, ~] = svd(data * data' / size(data, 2)) ; % ��Э��������svd�ֽ�
    data = sqrt(s) \ u' * data; % �׻���ȥ����ԣ�Э����Ϊ1��
end

function labels = load_MNIST_labels( file_name )
%����һ�� #��ǩ�� * #1 ��������

    %% ��ȡ raw MNIST labels
    fp = fopen(file_name, 'rb');
    assert(fp ~= -1, ['Could not open ', file_name, ' ' ]);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', file_name, ' ']);

    num_labels = fread(fp, 1, 'int32', 0, 'ieee-be');
    labels = fread(fp, inf, 'unsigned char');

    assert(size(labels, 1) == num_labels, 'Mismatch in label count');
    fclose(fp);

    labels(labels == 0) = 10;

    % ���汾�뻯�ɾ�����ʽ�ģ�������softmax��û����
    % index_row     = labels';
    % index_col     = 1:num_labels;
    % index         = (index_col - 1) .* 10 + index_row;
    % labels        = zeros(10, num_labels);
    % labels(index) = 1;
end



