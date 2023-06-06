function accuracy = get_accuracy_rate( predict_labels, labels )
%����Ԥ��׼ȷ��
% by ֣��ΰ Ewing 2016-04

% ��Ԥ��ĸ��ʾ����У�ÿ�������ʵ�ֵ��1��������0
predict_labels = bsxfun(@eq, predict_labels, max( predict_labels ));
predict_labels(:, sum(predict_labels)>1) = 0;  % ���ֵ�����Ӧ������0��Ҳ���ǲ���ȷ
% �ҳ���ȷlabel����Ӧ�����λ�ã�������Щλ�õ�ֵ���ֵ
index_row = labels';
index_col = 1:length(index_row);
index     = (index_col - 1) .* size( predict_labels, 1 ) + index_row;
accuracy  = sum( predict_labels(index) )/length(index_row);

end