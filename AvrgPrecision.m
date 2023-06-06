% Train_FM 训练数据全矩阵
% G 学习到的新矩阵：在该算法中，等于ksi2'*ksi1。
function avgPrecision = AvrgPrecision(Train_FM, All_FM, G)

    relevancThreshold = 0.8;
    Predicted = G;
    % 将全矩阵中为0的元素置为-1（这些用户商品对从一开始就没有值）；将训练集中出现的用户商品对对应的元素置为-1.
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
        % topk个元素即为预测的元素
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