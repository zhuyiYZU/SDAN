% by ֣��ΰ Ewing 2016-04
clc;clear
%% ��ȡ image �� label
[images_train, labels_train] = load_MNIST_data('dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'min_max_scaler', 0);
% images_train = images_train(:, 1:6000);
% labels_train = labels_train(1:6000, :);
[images_Test, labels_Test] = load_MNIST_data('dataSet/t10k-images.idx3-ubyte',...
    'dataSet/t10k-labels.idx1-ubyte', 'min_max_scaler', 0);

%% ���� SAEѵ��ʱ ����
architecture = [784 400 200 10]; % SAE����Ľṹ
% ���� AE��Ԥѡ���� �� BP��Ԥѡ����
preOption_SAE.option_AE.activation  = {'ReLU'};

preOption_SAE.option_AE.is_sparse    = 1;
preOption_SAE.option_AE.sparse_rho   = 0.01;
preOption_SAE.option_AE.sparse_beta  = 0.3;

preOption_SAE.option_AE.is_denoising = 1;
preOption_SAE.option_AE.noise_rate   = 0.15;
% preOption_SAE.option_AE.noise_layer = 'all_layers';
preOption_SAE.option_BP.activation = {'softmax'};
% �õ�SAE��Ԥѡ����
option_SAE = get_SAE_option(preOption_SAE);

%% ���� SAEԤ��ʱ ����
preOption_BPNN.activation = {'ReLU'; 'ReLU'; 'softmax'};
option_BPNN = get_BPNN_option(preOption_BPNN);

%% ���� runSAEOnce ����һ��SAE
is_disp_network = 0; % ��չʾ����
is_disp_info    = 0; % չʾ��Ϣ

[opt_theta, accuracy] = run_SAE_once(images_train, labels_train, ...
    images_Test, labels_Test, ... % ����
    architecture, ...
    option_SAE, option_BPNN, ...
    is_disp_network, is_disp_info);

% ����30�Σ���� ��ֵ����׼� ������95%����������
% accuracy = zeros(30, 1);
% for i = 1:30
%     [opt_theta, accuracy(i, 1)] = runSAEOnce(images_Train, labels_Train, ...
%         images_Test, labels_Test, ... % ����
%         architecture, ...
%         option_SAE, option_BPNN, ...
%         is_disp_network, is_disp_info);
%     
%     disp(['��' num2str(i) '�ε���']);
% end

% mean_accuracy = mean(accuracy);
% std_accuracy  = sqrt(sum((accuracy - mean_accuracy) .^ 2) / (size(accuracy, 1) - 1));
% up_bound      = mean_accuracy + 1.96 * std_accuracy;
% low_bound     = mean_accuracy - 1.96 * std_accuracy;
% disp(['���Ŷ� 95% ������£�׼ȷ��Ϊ�� ['...
%     num2str(low_bound) ',' num2str(up_bound) ']']);



