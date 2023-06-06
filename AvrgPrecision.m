% Train_FM ѵ������ȫ����
% G ѧϰ�����¾����ڸ��㷨�У�����ksi2'*ksi1��
function avgPrecision = AvrgPrecision(Train_FM, All_FM, G)

    relevancThreshold = 0.8;
    Predicted = G;
    % ��ȫ������Ϊ0��Ԫ����Ϊ-1����Щ�û���Ʒ�Դ�һ��ʼ��û��ֵ������ѵ�����г��ֵ��û���Ʒ�Զ�Ӧ��Ԫ����Ϊ-1.
    Predicted(find(All_FM == 0)) = -1;
    Predicted(find(Train_FM > 0)) = -1; 
%     All_FM(All_FM==0)=-1;
%     Train_FM(Train_FM>0)=-1;
    
    avgPEffectiveUserCount = 0;
    avgPrecision = 0;
    numU = size(Predicted, 1);
    for u = 1:numU
        tmp = Predicted(u, :);
        [stmp, index] = sort(tmp, 'descend');
        k = length(find(tmp>0));
        if k == 0
            continue
        end
        curRecommanded = zeros(1,k);
        curRelevance = zeros(1,k);
        relCount = 0;
        recCount = 0;
        % topk��Ԫ�ؼ�ΪԤ���Ԫ��
        for i = 1:k
            recCount = recCount+1;
            cur_predicted_index = index(1,i);
           if All_FM(u,cur_predicted_index) >= relevancThreshold
               relCount = relCount+1;
           end
           curRecommanded(1,i) = recCount;
           curRelevance(1,i) = relCount;
        end
        precisionSum = 0;
        relevantCount = 0;
        for i = 1:k
            cur_predicted_index = index(1,i);
           if  All_FM(u,cur_predicted_index) >= relevancThreshold && curRecommanded(1,i)>0
               relevantCount = relevantCount+1;
               precisionSum = precisionSum + curRelevance(1,i)/curRecommanded(1,i);
           end
        end
        if relevantCount>0
            avgPEffectiveUserCount = avgPEffectiveUserCount + 1;
            avgPrecision = avgPrecision + precisionSum/relevantCount;
        end
    end
    avgPrecision = avgPrecision/avgPEffectiveUserCount;
    
end