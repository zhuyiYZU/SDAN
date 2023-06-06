function [diff, num_gradient, grad] = check_BP(images, labels)
%���ڼ��sparseAutoencoderEpoch�������õ����ݶ�grad�Ƿ���Ч
% by ֣��ΰ Ewing 2016-04
% ��������ֵ�����ݶȵķ����õ��ݶ�numGradient����������
% ��sparseAutoencoderEpoch��������ѧ�����������õ����ݶȣ��ܿ죩���бȽ�
% �õ������ݶ�������ŷʽ�����С��Ӧ�÷ǳ�֮С�Ŷԣ�

image = images(:, 1:1);% ��Ϊ������������Բų�ȡһ�����������ͼ��theta��308308ά����
label = labels(1, 1);

architecture = [784 196 10]; % AE����Ľṹ: input_size -> hidden_size -> output_size
last_active_is_softmax = 1;
theta = init_parameters(architecture,...
    last_active_is_softmax); % ��������ṹ��ʼ���������

option.activation  = {'sigmoid', 'softmax'};
option.is_sparse    = 0;
option.sparse_rho   = 0.01;
option.sparse_beta  = 3;
option.is_denoising = 0;
option.decay_lambda = 1;

% ��������
[~, grad] = calc_BP_batch(image, label, theta, architecture, option);

% ��ֵ���㷽��
num_gradient = compute_numerical_gradient(...
    @(x) calc_BP_batch(image, label, x, architecture, option ), theta);

% �Ƚ��ݶȵ�ŷʽ����
diff = norm(num_gradient - grad) / norm(num_gradient + grad);

end






function num_gradient = compute_numerical_gradient(fun, theta)
%����ֵ�������� ����fun �� ��theta �����ݶ�
% fun��������theta�����ʵֵ�ĺ��� y = fun(theta)
% theta����������

    % ��ʼ�� num_gradient
    num_gradient = zeros(size(theta));

    % ��΢�ֵ�ԭ���������ݶȣ�����һ��С�仯�󣬺���ֵ�ñ仯�̶�
    EPSILON    = 1e-4;
    up_theta   = theta;
    down_theta = theta;
    
    wait = waitbar(0, '��ǰ����');
    for i = 1: length(theta)
        % waitbar( i/length(theta), wait, ['��ǰ����', num2str(i/length(theta)),'%'] );
        waitbar(i/length(theta), wait);
        
        up_theta(i)    = theta(i) + EPSILON;
        [result_up, ~] = fun(up_theta);
        
        down_theta(i)    = theta(i) - EPSILON;
        [result_down, ~] = fun(down_theta);
        
        num_gradient(i)  = (result_up - result_down) / ( 2 * EPSILON ); % d Vaule / d x
        
        up_theta(i)   = theta(i);
        down_theta(i) = theta(i);
    end
    bar  = findall(get(get(wait, 'children'), 'children'), 'type', 'patch');
    set(bar, 'facecolor', 'g');
    close(wait);
end
