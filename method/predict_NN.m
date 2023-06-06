function predict_labels = predict_NN(input, architecture, theta, option)
%��������ǰ��׶Σ���ʵ��Ԥ��
% by ֣��ΰ Ewing 2016-04

start_index = 1; % �洢�������±����
for i = 1:(length(architecture) - 1)
    visible_size = architecture(i);
    hidden_size  = architecture(i + 1);
    
    %% �Ƚ� theta ת��Ϊ (W, b) �ľ���/���� ��ʽ���Ա����������init_parameters�ļ����Ӧ��
    end_index = hidden_size * visible_size + start_index - 1; % �洢�������±��յ�
    W = reshape(theta(start_index : end_index), hidden_size, visible_size);
    
    if strcmp(option.activation{i}, 'softmax') % softmax����Ҫƫ��b
        start_index = end_index + 1; % �洢�������±����
    else
        start_index = end_index + 1; % �洢�������±����
        end_index = hidden_size + start_index - 1; % �洢�������±��յ�
        b = theta(start_index : end_index);
        start_index = end_index + 1;
    end
    
    %% feed forward �׶�
    activation_func = str2func(option.activation{i}); % �� ������� תΪ �����
    % �����ز�
    if strcmp(option.activation{i}, 'softmax') % softmax����Ҫƫ��b
        hidden_V = W * input; % ��� -> �յ��ֲ���V
    else
        hidden_V = bsxfun(@plus, W * input, b); % ��� -> �յ��ֲ���V
    end
    hidden_X = activation_func(hidden_V); % �����
    
    clear input
    input = hidden_X;
end

predict_labels = input;

end


%% �����
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));  
end
% tanh���Դ�����
function x = ReLU(x)
    x(x < 0) = 0;
end
function x = weakly_ReLU(x)
    x(x < 0) = x(x < 0) * 0.2;
end
function soft = softmax(x)
    soft = exp(x);
    soft = bsxfun(@rdivide, soft, sum(soft, 1));
end