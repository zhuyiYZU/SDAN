function option_SAE = get_SAE_option(preOption_SAE, varargin)
%����SAE�Ĳ���
% by ֣��ΰ Ewing 2016-04
% ���룺SAE�����ѡ�� preOption_SAE
% ���أ�SAE�����ѡ�� option_SAE

if exist('preOption_SAE', 'var')
    % �õ�AE��һЩԤѡ����
    if isfield(preOption_SAE, 'option_AE')
        option_SAE.option_AE = get_AE_option(preOption_SAE.option_AE); 
    else
        option_SAE.option_AE = get_AE_option([]);
    end
    % �õ�BP��һЩԤѡ����
    if isfield(preOption_SAE, 'option_BP')
        option_SAE.option_BP = get_BP_option(preOption_SAE.option_BP);
    else
        option_SAE.option_BP = get_BP_option([]);
    end
else
    option_SAE.option_AE = get_AE_option([]); % �õ�AE��һЩԤѡ����
    option_SAE.option_BP = get_BP_option([]); % �õ�BP��һЩԤѡ����
end

end


function option_AE = get_AE_option(preOption_AE)
%����AE�Ĳ���
% ���� AE�����ѡ�� preOption_AE
% ���أ�
% AE�����ѡ�option_AE
% decay_lambda��		Ȩ��˥��ϵ�������������Ȩ�أ�
% activation��		��������ͣ�sigmoid��ReLU��weakly_ReLU��tanh��������ͣ�sigmoid��ReLU��weakly_ReLU��tanh
% slope��			�����Ϊweakly_ReLUʱ���������б�ʣ�Ĭ��0.2��

% is_batch_norm��	�Ƿ�ʹ�� Batch Normalization �� speed-upѧϰ�ٶȣ�

% is_sparse��		�Ƿ�ʹ�� sparse hidden level �Ĺ���
% sparse_rho��		ϡ������rho��
% sparse_beta��		ϡ���Է���Ȩ�أ�

% is_denoising��		�Ƿ�ʹ�� denoising ����
% noise_layer��		AE����������Ĳ㣺'first_layer' or 'all_layers'
% noise_rate��		ÿһλ��������ĸ���
% noise_mode��		���������ģʽ��'On_Off' or 'Guass'
% noise_mean��		��˹ģʽ����ֵ
% noise_sigma��		��˹ģʽ����׼��

% is_weighted_cost��	�Ƿ��ÿһλ���ݵ�cost���м�Ȩ�Դ�
% weighted_cost��	��Ȩcost��Ȩ��

    if isfield(preOption_AE, 'decay_lambda')
        option_AE.decay_lambda = preOption_AE.decay_lambda;
    else
        option_AE.decay_lambda = 0.01;
    end
    if isfield(preOption_AE, 'activation')
        option_AE.activation = preOption_AE.activation;
		if strcmp(option_AE.activation{:}, 'weakly_ReLU')
			if isfield( preOption_AE, 'slope' )
				option_AE.slope = preOption_AE.slope;
			else
				option_AE.slope = 0.2;
			end
		end
    else
        option_AE.activation = { 'sigmoid' };
    end

    % batchNorm
    if isfield(preOption_AE, 'is_batch_norm')
        option_AE.is_batch_norm = preOption_AE.is_batch_norm;
    else
        option_AE.is_batch_norm = 0;
    end

    % sparse
    if isfield(preOption_AE, 'is_sparse')
        option_AE.is_sparse = preOption_AE.is_sparse;
    else
        option_AE.is_sparse = 0;
    end
    if option_AE.is_sparse
        if isfield(preOption_AE, 'sparse_rho')
            option_AE.sparse_rho = preOption_AE.sparse_rho;
        else
            option_AE.sparse_rho = 0.1;
        end
        if isfield(preOption_AE, 'sparse_beta')
            option_AE.sparse_beta = preOption_AE.sparse_beta;
        else
            option_AE.sparse_beta = 0.3;
        end
    end

    % de-noising
    if isfield(preOption_AE, 'is_denoising')
        option_AE.is_denoising = preOption_AE.is_denoising;
		if option_AE.is_denoising
			% de-noisingÿһ�� �� ֻ��һ�������
			if isfield(preOption_AE, 'noise_layer')
				option_AE.noise_layer = preOption_AE.noise_layer;
			else
				option_AE.noise_layer = 'first_layer';
			end
			% ��������
			if isfield(preOption_AE, 'noise_rate')
				option_AE.noise_rate = preOption_AE.noise_rate;
			else
				option_AE.noise_rate = 0.1;
			end
			% ����ģʽ����˹ �� ����
			if isfield(preOption_AE, 'noise_mode')
				option_AE.noise_mode = preOption_AE.noise_mode;
			else
				option_AE.noise_mode = 'On_Off';
			end
			switch option_AE.noise_mode
				case 'Guass'
					if isfield(preOption_AE, 'noise_mean')
						option_AE.noise_mean = preOption_AE.noise_mean;
					else
						option_AE.noise_mean = 0;
					end
					if isfield(preOption_AE, 'noise_sigma')
						option_AE.noise_sigma = preOption_AE.noise_sigma;
					else
						option_AE.noise_sigma = 0.01;
					end
			end
		end
    else
        option_AE.is_denoising = 0;
    end

    % weighted_cost
    if isfield(preOption_AE, 'is_weighted_cost')
        option_AE.is_weighted_cost = preOption_AE.is_weighted_cost;
    else
        option_AE.is_weighted_cost = 0;
    end
    if option_AE.is_weighted_cost
        if isfield(preOption_AE, 'weighted_cost')
            option_AE.weighted_cost = preOption_AE.weighted_cost;
%         else
%             error( '��Ȩcostһ��Ҫ�Լ�����Ȩ��������' );
        end
    end
end


function option_BP = get_BP_option(preOption_BP)
%����BP�Ĳ���
% ���� BP�����ѡ�� preOption_BP
% ���أ�
% AE�����ѡ�option_BP
% decay_lambda��	Ȩ��˥��ϵ�������������Ȩ�أ�
% activation��	��������ͣ�

% is_batch_norm���Ƿ�ʹ�� Batch Normalization �� speed-upѧϰ�ٶȣ�

% is_denoising��	�Ƿ�ʹ�� denoising ����
% noise_layer��	AE����������Ĳ㣺'first_layer' or 'all_layers'
% noise_rate��	ÿһλ��������ĸ���
% noise_mode��	���������ģʽ��'On_Off' or 'Guass'
% noise_mean��	��˹ģʽ����ֵ
% noise_sigma��	��˹ģʽ����׼��

    if isfield(preOption_BP, 'decay_lambda')
        option_BP.decay_lambda = preOption_BP.decay_lambda;
    else
        option_BP.decay_lambda = 0.001;
    end
    if isfield(preOption_BP, 'activation')
        option_BP.activation = preOption_BP.activation;
    else
        option_BP.activation = {'softmax'};
    end

    % batch normalization
    if isfield(preOption_BP, 'is_batch_norm')
        option_BP.is_batch_norm = preOption_BP.is_batch_norm;
    else
        option_BP.is_batch_norm = 0;
    end

    % de-noising
    if isfield(preOption_BP, 'is_denoising')
        option_BP.is_denoising = preOption_BP.is_denoising;
		if option_BP.is_denoising
			% denoisingÿһ�� �� ֻ��һ�������
			if isfield(preOption_BP, 'noise_layer')
				option_BP.noise_layer = preOption_BP.noise_layer;
			else
				option_BP.noise_layer = 'first_layer';
			end
			% ��������
			if isfield(preOption_BP, 'noise_rate')
				option_BP.noise_rate = preOption_BP.noise_rate;
			else
				option_BP.noise_rate = 0.1;
			end
			% ����ģʽ����˹ �� ����
			if isfield(preOption_BP, 'noise_mode')
				option_BP.noise_mode = preOption_BP.noise_mode;
			else
				option_BP.noise_mode = 'OnOff';
			end
			switch option_BP.noise_mode
				case 'Guass'
					if isfield(preOption_BP, 'noise_mean')
						option_BP.noise_mean = preOption_BP.noise_mean;
					else
						option_BP.noise_mean = 0;
					end
					if isfield(preOption_BP, 'noise_sigma')
						option_BP.noise_sigma = preOption_BP.noise_sigma;
					else
						option_BP.noise_sigma = 0.01;
					end
			end
		end
    else
        option_BP.is_denoising = 0;
    end
end




