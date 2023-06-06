function opt_theta = UDDSA(images_train, labels_train, images_Test, labels_Test,numX)
    %% 设置 训练时 参数
    featureNum = size(images_train,1);
    architecture = [featureNum 400 200 2]; % SAE网络的结构
    % 设置 AE的预选参数 及 BP的预选参数
    preOption_SAE.option_AE.activation  = {'sigmoid'};

    preOption_SAE.option_AE.is_sparse    = 1;
    preOption_SAE.option_AE.sparse_rho   = 0.01;
    preOption_SAE.option_AE.sparse_beta  = 0.3;

    preOption_SAE.option_AE.is_denoising = 1;
%     preOption_SAE.option_AE.noise_rate   = 0.15;
    preOption_SAE.option_AE.noise_rate   = 0.05;
    % preOption_SAE.option_AE.noise_layer = 'all_layers';
    preOption_SAE.option_BP.activation = {'softmax'};
    % 得到SAE的预选参数
    option_SAE = get_SAE_option(preOption_SAE);

    %% 设置 预测时 参数
    preOption_BPNN.activation = {'sigmoid'; 'sigmoid'; 'softmax'};
    option_BPNN = get_BPNN_option(preOption_BPNN);
    
   %% 设置 非平衡时 参数
   option_unbalance.numM = size(images_train,1);                % input data feature dimensions
   option_unbalance.numK = 10;   %80;%10;                       % number of hidden units 
   option_unbalance.numC = 2;                                   % number of classes
   option_unbalance.numX = numX;
   option_unbalance.alpha = 0.5;%1; 
   option_unbalance.beta = 0.5;%0.5;
   option_unbalance.gamma = 0.00001;%0.0001;

   %% 调用 run_UDDSA_once 运行一次SAE
   is_disp_network = 0; % 不展示网络
   is_disp_info    = 0; % 展示信息

   [opt_theta, accuracy] = run_UDDSA_once(images_train, labels_train, ...
        images_Test, labels_Test, ... % 数据
        architecture, ...
        option_SAE, option_BPNN, option_unbalance,...
        is_disp_network, is_disp_info);

    % 运行30次，求得 均值、标准差， 并计算95%的置信区间
    % accuracy = zeros(30, 1);
    % for i = 1:30
    %     [opt_theta, accuracy(i, 1)] = run_UDDSA_once(images_Train, labels_Train, ...
    %         images_Test, labels_Test, ... % 数据
    %         architecture, ...
    %         option_SAE, option_BPNN, ...
    %         is_disp_network, is_disp_info);
    %
    %     disp(['第' num2str(i) '次迭代']);
    % end

    % mean_accuracy = mean(accuracy);
    % std_accuracy  = sqrt(sum((accuracy - mean_accuracy) .^ 2) / (size(accuracy, 1) - 1));
    % up_bound      = mean_accuracy + 1.96 * std_accuracy;
    % low_bound     = mean_accuracy - 1.96 * std_accuracy;
    % disp(['置信度 95% 的情况下，准确率为： ['...
    %     num2str(low_bound) ',' num2str(up_bound) ']']);
end