%�����õ��ļ�
% by ֣��ΰ Ewing 2016-04

%% ��֤AE�ݶȼ������ȷ��
% diff = check_AE(images); % �Ѿ���֤��������ʱ�䣬�����У������������
% disp(diff); % diffӦ�ú�С

%% ���� sparse DAE��ѵ��һ�� sparse DAE�����ع�������ԭ���ݽ��жԱ� - DAEͨ��
clc;clear
% �õ� loadMNISTImages��getAEOption��initializeParameters��trainAE����
[input, labels] = load_MNIST_data( 'dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'min_max_scaler', 1 );
architecture = [784 196 784];
% ���� AE��Ԥѡ���� �� BP��Ԥѡ����
preOption_SAE.option_AE.is_sparse    = 1;
preOption_SAE.option_AE.is_denoising = 1;
preOption_SAE.option_AE.activation  = {'ReLU'};
% �õ�SAE��Ԥѡ����
option_SAE = get_SAE_option(preOption_SAE);
option_AE  = option_SAE.option_AE;

count_AE = 1;

theta = init_parameters(architecture);
[opt_theta, cost] = trainAE(input, theta, architecture, count_AE, option_AE);

% ��ѵ���õ�AE���ع�������ͼƬ�������ԭʼͼƬ���жԱ�
option_AE.activation = {'ReLU'; 'ReLU'};
predict = predict_NN( input, architecture, opt_theta, option_AE );

images_predict = reshape(predict, sqrt(size(predict, 1)), sqrt(size(predict, 1)), size(predict, 2));
% �Ҷ�ͼ
figure('NumberTitle', 'off', 'Name', 'MNIST��д����ͼƬ(�ع���');
show_images_num = 200;
penal           = show_images_num * 2 / 3;
pic_mat_col     = ceil(1.5 * sqrt(penal));
pic_mat_row     = ceil(show_images_num / pic_mat_col);
for i = 1:show_images_num
    subplot(pic_mat_row, pic_mat_col, i, 'align' );
    imshow(images_predict(:, :, i));
end
% ����ͼ jet
figure('NumberTitle', 'off', 'Name', 'MNIST��д����ͼƬ(�ع���-����ͼ');
for i = 1:show_images_num
    subplot(pic_mat_row, pic_mat_col, i, 'align');
    imagesc(images_predict(:, :, i));
    axis off;
end



