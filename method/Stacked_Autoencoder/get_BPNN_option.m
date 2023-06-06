function option_BPNN = get_BPNN_option(preOption_BPNN)
%����BP�Ĳ���
% by ֣��ΰ Ewing 2016-04
% ���� BP�����ѡ�� preOption_BPNN
% ���أ�
% AE�����ѡ� option_BPNN
% decay_lambda��   Ȩ��˥��ϵ�������������Ȩ�أ�
% activation��     ��������ͣ�

% is_batch_norm��  �Ƿ�ʹ�� Batch Normalization �� speed-upѧϰ�ٶȣ�

% is_denoising��   �Ƿ�ʹ�� denoising ����
% noise_layer��    AE����������Ĳ㣺'first_layer' or 'all_layers'
% noise_rate��     ÿһλ��������ĸ���
% noise_mode��     ���������ģʽ��'On_Off' or 'Guass'
% noise_mean��     ��˹ģʽ����ֵ
% noise_sigma��    ��˹ģʽ����׼��

if isfield(preOption_BPNN, 'decay_lambda')
	option_BPNN.decay_lambda = preOption_BPNN.decay_lambda;
else
	option_BPNN.decay_lambda = 0.001;
end

if isfield(preOption_BPNN, 'activation')
	option_BPNN.activation = preOption_BPNN.activation;
else
	error('������б���������Լ�������');
end

% batch normalization
if isfield(preOption_BPNN, 'is_batch_norm')
	option_BPNN.is_batch_norm = preOption_BPNN.is_batch_norm;
else
	option_BPNN.is_batch_norm = 0;
end

% de-noising
if isfield(option_BPNN, 'is_denoising')
    option_BPNN.is_denoising = option_BPNN.is_denoising;
    if option_BPNN.is_denoising
        % denoisingÿһ�� �� ֻ��һ�������
        if isfield(option_BPNN, 'noise_layer')
            option_BPNN.noise_layer = option_BPNN.noise_layer;
        else
            option_BPNN.noise_layer = 'first_layer';
        end
        % ��������
        if isfield(option_BPNN, 'noise_rate')
            option_BPNN.noise_rate = option_BPNN.noise_rate;
        else
            option_BPNN.noise_rate = 0.1;
        end
        % ����ģʽ����˹ �� ����
        if isfield(option_BPNN, 'noise_mode')
            option_BPNN.noise_mode = option_BPNN.noise_mode;
        else
            option_BPNN.noise_mode = 'On_Off';
        end
        switch option_BPNN.noise_mode
            case 'Guass'
                if isfield(option_BPNN, 'noise_mean')
                    option_BPNN.noise_mean = option_BPNN.noise_mean;
                else
                    option_BPNN.noise_mean = 0;
                end
                if isfield(option_BPNN, 'noise_sigma')
                    option_BPNN.noise_sigma = option_BPNN.noise_sigma;
                else
                    option_BPNN.noise_sigma = 0.01;
                end
        end
    end
else
    option_BPNN.is_denoising = 0;
end

end