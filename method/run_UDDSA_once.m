function [opt_theta, accuracy] = run_UDDSA_once(images_Train, labels_Train, ...
    images_Test, labels_Test, ... % ����
    architecture, ...
    option_SAE, option_BPNN, option_unbalance,...
    is_disp_network, is_disp_info)
    %����UDDSA���� �� ����һ�� SAE

%% ѵ��UDDSA
theta_SAE = train_UDDSA(images_Train, labels_Train, images_Test, labels_Test,architecture, option_SAE, option_unbalance); % ѵ��SAE
if is_disp_network
    % չʾ�����м������ȡ��feature
    display_network(reshape(theta_SAE(1 : 784 * 400), 400, 784));
    display_network((reshape(theta_SAE(1 : 784 * 400), 400, 784)' * ...
        reshape(theta_SAE(784 * 400 + 1 : 784 * 400 + 400*200 ), 200, 400)')');
end
if is_disp_info
    % �� δ΢����SAE���� ����Ԥ��
    predict_labels = predictNN(images_Train, architecture, theta_SAE, option_BPNN);
    accuracy = get_accuracy_rate(predict_labels, labels_Train);
    disp(['ѵ���� δ΢�� ׼ȷ��Ϊ�� ', num2str(accuracy * 100), '%']);
    
    predict_labels = predictNN( images_Test, architecture, theta_SAE, option_BPNN );
    accuracy = get_accuracy_rate( predict_labels, labels_Test );
    disp(['���Լ� δ΢�� ׼ȷ��Ϊ�� ', num2str(accuracy * 100), '%']);
end


%% BP fine-tune
[opt_theta, ~] = train_BPNN( images_Train, labels_Train, theta_SAE, architecture, option_BPNN );
if is_disp_network
    % չʾ�����м������ȡ��feature
    display_network(reshape(opt_theta(1 : 784 * 400), 400, 784) );
    display_network((reshape(opt_theta(1 : 400 * 784), 400, 784)' * ...
        reshape(opt_theta(784 * 400 + 1 : 784 * 400 + 400*200 ), 200, 400)')' );
end
%% �� fine-tune��SAE ����Ԥ��
if is_disp_info
    predict_labels = predict_NN(images_Train, architecture, opt_theta, option_BPNN);
    accuracy = get_accuracy_rate(predict_labels, labels_Train);
    disp(['ѵ���� ΢�� ׼ȷ��Ϊ�� ', num2str(accuracy * 100), '%']);
end
predict_labels = predict_NN( images_Train, architecture, opt_theta, option_BPNN);
accuracy = get_accuracy_rate( predict_labels, labels_Train );
disp(['ѵ���� ΢�� ׼ȷ��Ϊ�� ', num2str(accuracy * 100), '%'] );
predict_labels = predict_NN( images_Test, architecture, opt_theta, option_BPNN);
accuracy = get_accuracy_rate( predict_labels, labels_Test );
disp(['���Լ� ΢�� ׼ȷ��Ϊ�� ', num2str(accuracy * 100), '%']);

accuracy_LR = test_LR(images_Train, images_Test, labels_Train', labels_Test');
disp(['���Լ� LR������ ΢�� ׼ȷ��Ϊ�� ', num2str(accuracy_LR(2,1) * 100), '%']);

if is_disp_info
    disp(['���Լ� ΢�� ׼ȷ��Ϊ�� ', num2str(accuracy * 100), '%']);
end

end


