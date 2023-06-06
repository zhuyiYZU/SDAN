function display_network( weight, figure_name, ~ )
%��������Ȩ��(hidden_size*input_size)չʾ��������ȡ������ͼ
% ����ÿ�� hidden level 1 �� neuron ��ʾ����ȡ��һ�� feature
% �����ӵ� neuron A ��Ȩ������������ input vector ��ÿһλ�� feature A ����Ҫ�̶�
% ����Ȩ����������Ҫ�̶ȣ������ɹ���� input �� feature
% by ֣��ΰ Ewing 2016-04

% �� ÿ��inputλȨ�� ʵʩ��һ��
weight_min = min(weight, [], 2);
weight     = bsxfun(@minus, weight, weight_min);
weight_max = max( weight, [], 2 );
weight     = bsxfun(@rdivide, weight, weight_max);

feature_num   = size(weight, 1); % feature������Ҳ��ͼƬ����
penal         = feature_num * 2 / 3;
pic_mat_col   = ceil(1.5 * sqrt(penal));
pic_mat_row   = ceil(feature_num / pic_mat_col);

images = reshape(weight', sqrt(size(weight, 2)), sqrt(size(weight, 2)), feature_num); % ͼƬ
% չʾ����
% �Ҷ�ͼ
if exist('figure_name', 'var')
    figure('NumberTitle', 'off', 'Name', figure_name);
else
    figure('NumberTitle', 'off', 'Name', 'MNIST��д��������ͼ');
end
for i = 1:feature_num
    subplot( pic_mat_row, pic_mat_col, i, 'align' );
    imshow( images(:, :, i) );
%     imagesc( images(:, :, i) );
%     axis off;
end

end