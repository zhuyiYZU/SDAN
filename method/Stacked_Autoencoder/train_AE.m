function [opt_theta, cost] = train_AE(input, theta, architecture, count_AE, option_AE)
%ѵ��AE����
% by ֣��ΰ Ewing 2016-04

% ���� calc_AE_Batch ���Ը��ݵ�ǰ����� cost �� gradient�����ǲ�����ȷ��
% �������Mark Schmidt�İ����Ż����� ����������l-BFGS
% Mark Schmidt (http://www.di.ens.fr/~mschmidt/Software/minFunc.html) [����ѧ��]
addpath minFunc/
options.Method = 'lbfgs'; % ��ʵ��һ����L-BFGS�����Բο� On optimization methods for deep learning
options.maxIter = 100;	  % L-BFGS ������������
options.display = 'off';
% options.TolX = 1e-3;

% �жϸ� countAE�� AE�Ƿ���Ҫ���noise �� ʹ��denoising����
[is_denoising, input_corrupted ] = denoising_switch(input, count_AE, option_AE);
if is_denoising
	[opt_theta, cost] = minFunc(@(x) calc_AE_batch(input, x, architecture, option_AE, input_corrupted), ...
            theta, options);
else
	[opt_theta, cost] = minFunc(@(x) calc_AE_batch(input, x, architecture, option_AE), ...
            theta, options);
end

end

function [is_denoising, input_corrupted] = denoising_switch(input, count_AE, option_AE)
%�жϸò�AE�Ƿ���Ҫ���noise��ʹ��denoising����
% ���� �Ƿ�is_denoising�ı�־ �� ����

% is_denoising��	�Ƿ�ʹ�� denoising ����
% noise_layer��	AE����������Ĳ㣺'first_layer' or 'all_layers'
% noise_rate��	ÿһλ��������ĸ���
% noise_mode��	���������ģʽ��'On_Off' or 'Guass'
% noise_mean��	��˹ģʽ����ֵ
% noise_sigma��	��˹ģʽ����׼��

    is_denoising    = 0;
    input_corrupted = [];
    if option_AE.is_denoising
        switch option_AE.noise_layer
            case 'first_layer'
                if count_AE == 1
                    is_denoising = 1;
                end
            case 'all_layers'
                is_denoising = 1;
            otherwise
                error( '�����AE����������' );
        end
        
        if is_denoising
            input_corrupted = input;
            index_corrupted = rand(size(input)) < option_AE.noise_rate;
            switch option_AE.noise_mode
                case 'Guass'
                    % ��ֵΪ noise_mean����׼��Ϊ noise_sigma �ĸ�˹����
                    noise = option_AE.noise_mean + ...
                        randn(size(input)) * option_AE.noise_sigma;
                    noise(~index_corrupted) = 0;
                    input_corrupted = input_corrupted + noise;
                case 'On_Off'
                    input_corrupted(index_corrupted) = 0;
            end
        end
    end
end

