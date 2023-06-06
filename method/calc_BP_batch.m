function [cost,grad] = calc_BP_batch(input, output, theta, architecture, option)
%���� BPNN ���ݶȱ仯�����
% by ֣��ΰ Ewing 2016-04
% input��       ѵ����������ÿһ�д���һ��������
% theta��       Ȩֵ��������[ W1(:); b1(:); W2(:); b2(:); ... ]��
% architecture: ����ṹ��ÿ�������ɵ�������
% �ṹ�� option
% decay_lambda��      Ȩ��˥��ϵ�������������Ȩ�أ�
% activation��  ��������ͣ�

% is_batch_norm��   �Ƿ�ʹ�� Batch Normalization �� speed-upѧϰ�ٶȣ�

% ����ȷʹ��BP�Ĺ���
% option.is_batch_norm���ù���Ŀǰ��û��


m                = size(input, 2); % ������
layers           = length(architecture); % �������
% ��ʼ��һЩ����
layer_hidden_V     = cell(1, layers - 1); % ����ʢװÿһ����������յ��ֲ�������
layer_hidden_X     = cell(1, layers);     % ����ʢװÿһ������������/��������
layer_hidden_X{1}  = input;
cost_regul         = 0; % ������ķ�����
cost_error         = 0; % cost function
grad               = zeros(size(theta));
%% feed-forward�׶�
start_index = 1; % �洢�������±����
for i = 1:(layers - 1)
    visible_size = architecture(i);
    hidden_size  = architecture(i + 1);
    
    activation_func = str2func(option.activation{i}); % �� ������� תΪ �����
    
    % �Ƚ� theta ת��Ϊ (W, b) �ľ���/���� ��ʽ���Ա����������initializeParameters�ļ����Ӧ��
    end_index  = hidden_size * visible_size + start_index - 1; % �洢�������±��յ�
    W          = reshape(theta(start_index : end_index), hidden_size, visible_size);
    
    if strcmp(option.activation{i}, 'softmax') % softmax��һ�㲻��ƫ��b
        start_index = end_index + 1; % �洢�������±����
        
        hidden_V = W * input;% ��� -> �õ��յ��ֲ��� V
    else
        start_index = end_index + 1; % �洢�������±����
        end_index   = hidden_size + start_index - 1; % �洢�������±��յ�
        b           = theta(start_index : end_index);
        start_index = end_index + 1;
        
        hidden_V = bsxfun(@plus, W * input, b); % ��� -> �õ��յ��ֲ��� V
    end
    hidden_X = activation_func(hidden_V); % �����
    % ����������ķ�����
    cost_regul = cost_regul + 0.5 * option.decay_lambda * sum(sum(W .^ 2));
    
    clear input
    input = hidden_X;
    
    layer_hidden_V{i}     = hidden_V; % ����ʢװÿһ����������յ��ֲ�������
    layer_hidden_X{i + 1} = input;   % ����ʢװÿһ������������/��������
end
% ��cost function + regularization
if strcmp(option.activation{end}, 'softmax') % ��ǩ��cost
    % softmax��cost�����Ҳ�û������������Ҽ���1. ����ģ��׼ȷ��
    index_row = output';
    index_col = 1:m;
    index    = (index_col - 1) .* architecture(end) + index_row;
%     cost_error = sum(1 - layer_hidden_X{layers}(index)) / m;
    cost_error = - sum(log(layer_hidden_X{layers}(index))) / m; 
else % ʵֵ��cost
    cost_error = sum(sum((output - layer_hidden_X{layers}).^2)) ./ 2 / m;
end

cost = cost_error + cost_regul;

%% Back Propagation �׶Σ���ʽ������
% �����һ��
activation_func_deriv = str2func([option.activation{layers-1}, '_deriv']);
if strcmp(option.activation{layers-1}, 'softmax') % softmax��һ������Ҫ����labels��Ϣ
    dError_dOutputV   = activation_func_deriv(layer_hidden_V{layers - 1}, output);
else
    % dError/dOutputV = dError/dOutputX * dOutputX/dOutputV
    dError_dOutputX   = -(output - layer_hidden_X{layers});
    dOutputX_dOutputV = activation_func_deriv(layer_hidden_V{layers - 1});
    dError_dOutputV   = dError_dOutputX .* dOutputX_dOutputV;
end


% dError/dW = dError/dOutputV * dOutputV/dW
dOutputV_dW = layer_hidden_X{layers - 1}';
dError_dW   = dError_dOutputV * dOutputV_dW;

if strcmp(option.activation{layers-1}, 'softmax') % softmax��һ�㲻��ƫ��b
    end_index   = length(theta); % �洢�������±��յ�
    start_index = end_index + 1; % �洢�������±����
else
    % �����ݶ� b
    end_index   = length(theta); % �洢�������±��յ�
    start_index = end_index - architecture(end)  + 1; % �洢�������±����
    dError_db  = sum(dError_dOutputV, 2);
    grad(start_index:end_index) = dError_db ./ m;
end
% �����ݶ� W
end_index   = start_index - 1; % �洢�������±��յ�
start_index = end_index - architecture(end - 1) * architecture(end)  + 1; % �洢�������±����
W          = reshape(theta(start_index:end_index), architecture(end), architecture(end - 1 ));
W_grad      = dError_dW ./ m + option.decay_lambda * W;
grad(start_index:end_index) = W_grad(:);

% ���ش� error back-propagation
for i = (layers - 2):-1:1
    activation_func_deriv = str2func([option.activation{i}, '_deriv']);
    % dError/dHiddenV = dError/dHiddenX * dHiddenX/dHiddenV
    dError_dHiddenX   = W' * dError_dOutputV; % = dError/dOutputV * dOutputV/dHiddenX
    dHiddenX_dHiddenV = activation_func_deriv(layer_hidden_V{i});
    dError_dHiddenV   = dError_dHiddenX .* dHiddenX_dHiddenV;
    % dError/dW1 = dError/dHiddenV * dHiddenV/dW1
    dHiddenV_dW = layer_hidden_X{i}';
    dError_dW   = dError_dHiddenV * dHiddenV_dW;
    
    dError_db = sum(dError_dHiddenV, 2);
    % �����ݶ� b
    end_index   = start_index - 1; % �洢�������±��յ�
    start_index = end_index - architecture(i + 1)  + 1; % �洢�������±����
    % b          = theta( startIndex : endIndex );
    grad(start_index:end_index) = dError_db ./ m;
    
    % �����ݶ� W
    end_index   = start_index - 1; % �洢�������±��յ�
    start_index = end_index - architecture( i ) * architecture( i + 1 )  + 1; % �洢�������±����
    W          = reshape( theta(start_index:end_index), architecture( i + 1 ), architecture( i ) );
    W_grad      = dError_dW ./ m + option.decay_lambda * W;
    grad(start_index:end_index) = W_grad(:);
    
    dError_dOutputV = dError_dHiddenV;
end

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
%% ���������
function sigm_deriv = sigmoid_deriv(x)
    sigm_deriv = sigmoid(x).*(1-sigmoid(x));  
end
function tan_deriv = tanh_deriv(x)
    tan_deriv = 1 ./ cosh(x).^2; % tanh�ĵ���
end
function x = ReLU_deriv(x)
    x(x < 0) = 0;
    x(x > 0) = 1;
end
function x = weakly_ReLU_deriv(x)
    x(x < 0) = 0.2;
    x(x > 0) = 1;
end
function soft_deriv = softmax_deriv( x, labels )
    index_row = labels';
    index_col = 1:length(index_row);
    index    = (index_col - 1) .* max(labels) + index_row;
    
%     softDeriv = softmax(x);
%     active   = zeros( size(x) );
%     active(index) = 1;
%     softDeriv = bsxfun( @times, softDeriv - active, softDeriv(index) );

    soft_deriv = softmax(x);
    soft_deriv(index) = soft_deriv(index) - 1;  % �����ʹ��ԭʼcost function�ĵ���
end







