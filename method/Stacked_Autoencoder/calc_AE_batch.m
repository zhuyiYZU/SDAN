function [cost,grad] = calc_AE_batch( input, theta, architecture, option_AE, input_corrupted, ~ )
%����ϡ���Ա��������ݶȱ仯�����
% by ֣��ΰ Ewing 2016-04
% input��       ѵ����������ÿһ�д���һ��������
% theta��       Ȩֵ��������[ W1(:); b1(:); W2(:); b2(:); ... ]��
% architecture: ����ṹ��ÿ�������ɵ�������
% �ṹ�� option_AE
% decay_lambda�� Ȩ��˥��ϵ�������������Ȩ�أ�
% activation��  ��������ͣ�

% is_batch_norm�� �Ƿ�ʹ�� Batch Normalization �� speed-upѧϰ�ٶȣ�

% is_sparse��    �Ƿ�ʹ�� sparse hidden level �Ĺ���
% sparse_rho��   ϡ������rho��һ�㸳ֵΪ 0.01��
% sparse_beta��  ϡ���Է���Ȩ�أ�

% input_corrupted�� ʹ�� denoising ���� ���иò�������

% ����ȷʹ��AE�Ĺ���
% option_AE.is_batch_norm���ù���Ŀǰ��û��

visible_size = architecture(1);
hidden_size  = architecture(2);
% �Ƚ� theta ת��Ϊ (W1, W2, b1, b2) �ľ���/���� ��ʽ���Ա����������initializeParameters�ļ����Ӧ��
W1 = reshape(theta(1 : (hidden_size * visible_size)), ...
    hidden_size, visible_size);
b1 = theta((hidden_size * visible_size + 1) : (hidden_size * visible_size + hidden_size));
W2 = reshape(theta((hidden_size * visible_size + hidden_size + 1) : (2 * hidden_size * visible_size + hidden_size)), ...
    visible_size, hidden_size);
b2 = theta((2 * hidden_size * visible_size + hidden_size + 1) : end);

m = size(input, 2); % ������

%% feed forward �׶�
activation_func = str2func(option_AE.activation{:}); % �� ������� תΪ �����
% �����ز�
if exist('input_corrupted', 'var')
	hidden_V = bsxfun(@plus, W1 * input_corrupted, b1); % ��� -> V
else
	hidden_V = bsxfun(@plus, W1 * input, b1); % ��� -> V
end
hidden_X = activation_func(hidden_V); % �����

% �������ز��ϡ�跣��
if option_AE.is_sparse
    rho_hat = sum(hidden_X, 2) / m;
    KL     = get_KL(option_AE.sparse_rho, rho_hat);
    cost_sparse = option_AE.sparse_beta * sum(KL);
else
    cost_sparse = 0;
end

% �������
output_V = bsxfun(@plus, W2 * hidden_X, b2); % ��� -> V
output_X = activation_func(output_V);   % �����
  
% ��cost function + regularization
cost_error = sum(sum((output_X - input).^2)) / m / 2;
cost_regul = 0.5 * option_AE.decay_lambda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));  

% ���ܵ�cost
cost = cost_error + cost_regul + cost_sparse;

%% Back Propagation �׶�
activation_func_deriv = str2func([option_AE.activation{:}, '_deriv']);
% ��ʽ������
% dError/dOutputV = dError/dOutputX * dOutputX/dOutputV
dError_dOutputX   = -(input - output_X);
dOutputX_dOutputV = activation_func_deriv(output_V);
dError_dOutputV   = dError_dOutputX .* dOutputX_dOutputV;
% dError/dW2 = dError/dOutputV * dOutputV/dW2
dOutputV_dW2 = hidden_X';
dError_dW2   = dError_dOutputV * dOutputV_dW2;

W2_grad       = dError_dW2 ./ m + option_AE.decay_lambda * W2;
% dError/dHiddenV = (dError/dHiddenX + dSparse/dHiddenX) * dHiddenX/dHiddenV
dError_dHiddenX   = W2' * dError_dOutputV; % = dError/dOutputV * dOutputV/dHiddenX
dHiddenX_dHiddenV = activation_func_deriv(hidden_V);
if option_AE.is_sparse
    dSparse_dHiddenX = option_AE.sparse_beta .* get_KL_deriv(option_AE.sparse_rho, rho_hat);
    dError_dHiddenV  = (dError_dHiddenX + repmat(dSparse_dHiddenX, 1, m)) .* dHiddenX_dHiddenV;
else
    dError_dHiddenV  = dError_dHiddenX .* dHiddenX_dHiddenV;
end
% dError/dW1 = dError/dHiddenV * dHiddenV/dW1
dHiddenV_dW1 = input';
dError_dW1   = dError_dHiddenV * dHiddenV_dW1;

W1_grad       = dError_dW1 ./ m + option_AE.decay_lambda * W1;


% ���ڽ����ݶ���ʧ������������
% disp('�ݶ���ʧ');
% disp(['W2�ݶȾ���ֵ��ֵ��', num2str(mean(mean(abs(W2_grad)))), ...
%     ' -> ','W1�ݶȾ���ֵ��ֵ��', num2str(mean(mean(abs(W1_grad))))]);
% disp(['W2�ݶ����ֵ��', num2str(max(mean(W2_grad))), ...
%     ' -> ','W1�ݶ����ֵ��', num2str(max(mean(W1_grad)))]);


% ��ƫ�õĵ���
dError_db2 = sum(dError_dOutputV, 2);
b2_grad     = dError_db2 ./ m;
dError_db1 = sum(dError_dHiddenV, 2);  
b1_grad     = dError_db1 ./ m;

grad = [W1_grad(:); b1_grad(:); W2_grad(:); b2_grad(:)];

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

%% KLɢ�Ⱥ���������
function KL = get_KL(sparse_rho,rho_hat)
%KL-ɢ�Ⱥ���
    EPSILON = 1e-8; %��ֹ��0
    KL = sparse_rho .* log( sparse_rho ./ (rho_hat + EPSILON) ) + ...
        ( 1 - sparse_rho ) .* log( (1 - sparse_rho) ./ (1 - rho_hat + EPSILON) );  
end

function KL_deriv = get_KL_deriv(sparse_rho,rho_hat)
%KL-ɢ�Ⱥ����ĵ���
    EPSILON = 1e-8; %��ֹ��0
    KL_deriv = ( -sparse_rho ) ./ ( rho_hat + EPSILON ) + ...
        ( 1 - sparse_rho ) ./ ( 1 - rho_hat + EPSILON );  
end