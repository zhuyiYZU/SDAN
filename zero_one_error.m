% TCM 测试胞元矩阵，每一个胞元存放一个用户的数据，和CM格式一样，第一行是商品的index，第二行是商品的评分；
% ACM 胞元矩阵，格式同TCM；包含了所有的训练和测试数据；
% G 学习到的新矩阵：在该算法中，等于ksi2'*ksi1。
function loss = zero_one_error(CM, G)
    load('RatingMatrix_ACM');
    numU = size(CM, 1);
    loss = 0;
    % 对每个用户
    for u = 1:numU
        % vt该用户的测试数据，va该用户所有的数据
        vt = CM{u,1};
        va = ACM{u,1};
        num1 = size(vt,2);
        num2 = size(va,2);
        if num2 < 2
            continue
        end
        tloss = 0;
        for i = 1:num1
           for j = 1:num2
               % 同一个商品不跟自己比较，评分相同的商品不进行比较
               if vt(1,i) == va(1,j) || vt(2,i) == va(2,j)
                  continue
               end
               if vt(2,i)> va(2,j)
                   value = 1;
               elseif vt(2,i) < va(2,j)
                       value = -1;
               end
               if G(u,vt(1,i)) < G(u,va(1,j))
                   value = -value;
               end
               if value < 0
                  tloss = tloss + 1; 
               end
           end
        end
        loss = loss + tloss/(num1*(num2-1));
    end
    loss = loss/numU;
end