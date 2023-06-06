% TCM ���԰�Ԫ����ÿһ����Ԫ���һ���û������ݣ���CM��ʽһ������һ������Ʒ��index���ڶ�������Ʒ�����֣�
% ACM ��Ԫ���󣬸�ʽͬTCM�����������е�ѵ���Ͳ������ݣ�
% G ѧϰ�����¾����ڸ��㷨�У�����ksi2'*ksi1��
function loss = zero_one_error(CM, G)
    load('RatingMatrix_ACM');
    numU = size(CM, 1);
    loss = 0;
    % ��ÿ���û�
    for u = 1:numU
        % vt���û��Ĳ������ݣ�va���û����е�����
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
               % ͬһ����Ʒ�����Լ��Ƚϣ�������ͬ����Ʒ�����бȽ�
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