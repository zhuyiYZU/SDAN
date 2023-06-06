function theta = init_parameters(architecture, last_active_is_softmax, varargin )
%����ÿһ����Ԫ�����������ʼ��������Ȩ�ز���
% by ֣��ΰ Ewing 2016-04
% architecture: ����ṹ��
% theta��Ȩֵ��������[ W1(:); b1(:); W2(:); b2(:); ... ]��

% û�д��� last_active_is_softmax��Ĭ�ϲ��� softmax�����
if nargin == 1
    last_active_is_softmax = 0;
end
% �������������W������b����������ʼ����
if last_active_is_softmax % softmax��һ�㲻��ƫ��b
    count_W = architecture * [ architecture(2:end) 0 ]';
    count_B = sum(architecture(2:(end-1)));
    theta = zeros(count_W + count_B, 1);
else
    count_W = architecture * [architecture(2:end) 0]';
    count_B = sum(architecture(2:end));
    theta = zeros(count_W + count_B, 1);
end

% ���� Hugo Larochelle���� ��ʼ��ÿ������� W
start_index = 1; % ����ÿ������w���±����
for layer = 2:length(architecture)
    % ����ÿ������W���±��յ�
    end_index = start_index + ...
        architecture(layer)*architecture(layer -1) - 1;
    
    % Ȩ�س�ʼ����Χ��Hugo Larochelle����
    r = sqrt(6) / sqrt(architecture(layer) + architecture(layer -1));  
    
    % (layer -1)  -> layer, f( Wx + b )
    theta(start_index:end_index) = rand(architecture(layer) * architecture(layer -1), 1) * 2 * r - r;
    
    % ������һ������W���±���㣨����b��
    start_index = end_index + architecture(layer) + 1;
end

end