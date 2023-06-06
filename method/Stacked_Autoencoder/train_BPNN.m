function [opt_theta, cost] = train_BPNN(input, output, theta, architecture, option_BP)
%ѵ��BP����
% by ֣��ΰ Ewing 2016-04

% ���� calc_BP_batch ���Ը��ݵ�ǰ����� cost �� gradient�����ǲ�����ȷ��
% �������Mark Schmidt�İ����Ż����� ����������l-BFGS
% Mark Schmidt (http://www.di.ens.fr/~mschmidt/Software/minFunc.html) [����ѧ��]
addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = 100;	  % L-BFGS ������������
options.display = 'off';
% options.TolX = 1e-3;

[opt_theta, cost] = minFunc(@(x) calc_BP_batch(input, output, x, architecture, option_BP), ...
    theta, options);

end