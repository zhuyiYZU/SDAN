function theta_SAE = train_SAE(input, output, architecture, option)
%ѵ��Stacked AE
% by ֣��ΰ Ewing 2016-04

option_AE = option.option_AE; % �õ�AE��һЩԤѡ����
option_BP = option.option_BP; % �õ�BP��һЩԤѡ����

% ��ʼ��������� theta4SAE�����ڴ洢�ѵ�����������Ĳ���
if strcmp(option_BP.activation, 'softmax') % softmax��һ�㲻��ƫ��b
    count_W = architecture * [architecture(2:end) 0]';
    count_B = sum(architecture(2:(end - 1)));
    theta_SAE = zeros(count_W + count_B, 1);
else
    count_W = architecture * [architecture(2:end) 0]';
    count_B = sum(architecture(2:end));
    theta_SAE = zeros(count_W + count_B, 1);
end

%% ���AE���� architecture ѵ��
start_index = 1; % �洢�������±����
for count_AE = 1 : (length(architecture) - 2) % �����������BPѵ��
    % AE����Ľṹ: inputSize -> hiddenSize -> outputSize
    architecture_AE = ...
        [architecture(count_AE) ...
        architecture(count_AE + 1) ...
        architecture(count_AE)];
    theta_AE  = init_parameters(architecture_AE); % ��������ṹ��ʼ���������
    
    [opt_theta, cost] = train_AE(input, theta_AE, architecture_AE, count_AE, option_AE);
%     if count_AE == 1 % ���Ը���cost��������ж��Ƿ���Ҫ����ѵ��
%         [ opt_theta, cost ] = train_AE( input, opt_theta, architecture_AE, option_AE );
%     end
    
    disp(['��' num2str(count_AE) '��AE "' ...
        num2str(architecture_AE) '" ��ѵ������ǣ�'...
        num2str(cost)]);
    
    % �洢 AE��W1��b1 �� SAE ��
    end_index = architecture(count_AE) * architecture(count_AE + 1) + ...
        architecture(count_AE + 1) + start_index - 1;% �洢�������±��յ�
    theta_SAE(start_index : end_index) = opt_theta(1 : ...
        (architecture(count_AE) * architecture(count_AE + 1) + architecture(count_AE + 1)));
    
    % �޸�inputΪ��һ���output
    clear predict theta_AE opt_theta cost
    predict = predict_NN(input, architecture_AE(1:2),...
        theta_SAE( start_index : end_index ), option_AE);
    input = predict;
    
    start_index = end_index + 1;
end

%% BP��ѵ���������
architecture_BP = [architecture(end-1) architecture(end)]; % ���� BP ����ṹ
% ��������ṹ��ʼ�� BP�������
if strcmp(option_BP.activation, 'softmax') % softmax��һ�㲻��ƫ��b
    last_active_is_softmax = 1;
    theta_BP = init_parameters(architecture_BP, last_active_is_softmax);
else
    theta_BP = init_parameters(architecture_BP);
end

[opt_theta, cost] = train_BPNN(input, output, theta_BP, architecture_BP, option_BP); % ѵ��BP����
disp(['���һ��BP "' num2str(architecture_BP) '" ��ѵ������ǣ�' num2str(cost)]);

theta_SAE(start_index : end) = opt_theta;
    
end