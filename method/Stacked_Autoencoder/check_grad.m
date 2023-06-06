clc, clear;
% by ֣��ΰ Ewing 2016-04

% ��������
[ images_train, labels_train ] = load_MNIST_data( 'dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'min_max_scaler', 0 );

% ��� ����AE�ݶȵ�׼ȷ��
[diff, num_gradient, grad] = check_AE(images_train);
fprintf(['AE�м����ݶȵķ�����������ֵ�����Ĳ����ԣ�'...
    num2str(mean(abs(num_gradient - grad)))...
    ' �� ' num2str(diff) '\n']);

% ��� ����BP�ݶȵ�׼ȷ��
[diff, num_gradient, grad] = check_BP(images_train, labels_train);
fprintf(['AE�м����ݶȵķ�����������ֵ�����Ĳ����ԣ�'...
    num2str(mean(abs(num_gradient - grad)))...
    ' �� ' num2str(diff) '\n']);