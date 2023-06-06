function [object,grad] = computeObjAndGrad(weight, k, alpha, gama, II, Train_FM, Train_CM, All_FM, fid, isTest)

    global UC
    global IC
    
    %% 参数还原
    index = 0;
    W11 = reshape(weight(index+1:index+k*UC),k,UC);
    b11 = weight(index+k*UC+1:index+k*UC+k);
    W12 = reshape(weight(index+k*UC+k+1:index+2*k*UC+k),UC,k);
    b12 = weight(index+2*k*UC+k+1:index+2*k*UC+k+UC);
    index = index + 2*k*UC + k + UC;
    W21 = reshape(weight(index+1:index+k*IC),k,IC);
    b21 = weight(index+k*IC+1:index+k*IC+k);
    W22 = reshape(weight(index+k*IC+k+1:index+2*k*IC+k),IC,k);
    b22 = weight(index+2*k*IC+k+1:index+2*k*IC+k+IC);
    
    Train_FMT = Train_FM';
    %% 误差项  第一项
    ksi1 = sigmoid(W11*Train_FM + repmat(b11,1,IC));
    ksi2 = sigmoid(W21*Train_FMT+ repmat(b21,1,UC));
    M = ksi2'*ksi1;
    
    numU = size(Train_CM,1);
    loss = 0;
    grad_ksi1 = zeros(k, IC);
    grad_ksi2 = zeros(k, UC);
    for u = 1:numU
        tmp = Train_CM{u,1};
        uloss = 0;
        numI = size(tmp, 2);
        su = 0;
        tgrad1 = zeros(k, IC);
        tgrad2 = zeros(k, UC);
        
        i = 1:(numI-1);
        j = i + 1;
        p1 = tmp(2,i);
        p2 = tmp(2,j);
        d1 = tril(ones(numI-1,1) * p1 - p2' * ones(1,numI-1));%1*numI-1
        q1 = M(u,tmp(1,i));
        q2 = M(u,tmp(1,j));
        d2 = tril(ones(numI-1,1) * q1 - q2' * ones(1,numI-1));%1*numI-1
        
        Id = find(d1>0);
        [i j] = find(d1>0);
        if isempty(Id) == 0
        loss1 = d1(Id) .* log(1 + exp(-d2(Id)));
        s = - d1(Id) .* exp(-d2(Id)) ./ (1+exp(-d2(Id)));%sf * 1
        tgrad2(:,u) = tgrad2(:,u) + (ksi1(:,tmp(1,i)) - ksi1(:,tmp(1,j))) * s;
        tgrad1(:,i) = tgrad1(:,i) + ksi2(:,u) * sum(s) * ones(1,length(i));
        tgrad1(:,j) = tgrad1(:,j) - ksi2(:,u) * sum(s) * ones(1,length(i));
        uloss = uloss + sum(loss1);
        su = su + length(Id);
        end
        
        Id = find(d1<0);
        [i j] = find(d1<0);
        if isempty(Id) == 0
        loss2 = -d1(Id) .* log(1 + exp(d2(Id)));
        s = - d1(Id) .* exp(d2(Id)) ./ (1+exp(d2(Id)));
        tgrad2(:,u) = tgrad2(:,u) + (ksi1(:,tmp(1,i)) - ksi1(:,tmp(1,j))) * s;
        tgrad1(:,i) = tgrad1(:,i) + ksi2(:,u) * sum(s) * ones(1,length(i));
        tgrad1(:,j) = tgrad1(:,j) - ksi2(:,u) * sum(s) * ones(1,length(i));
        uloss = uloss + sum(loss2);
        su = su + length(Id);
        end
        
%         for i = 1:numI-1
%            for j = i+1:numI
%              if tmp(2,i) == tmp(2,j)
%                  continue;
%              end
%              su = su + 1;
%              % 目标值
%              uloss = uloss + getPenalty(tmp(2,i), tmp(2,j), M(u,tmp(1,i)), M(u,tmp(1,j)));
%              % 梯度
%              s = getPenaltyGrad(tmp(2,i), tmp(2,j), M(u,tmp(1,i)), M(u,tmp(1,j)));
%              tgrad2(:,u) = tgrad2(:,u) + s*(ksi1(:,tmp(1,i)) - ksi1(:,tmp(1,j)));
%              tgrad1(:,i) =  tgrad1(:,i) + s*ksi2(:,u);
%              tgrad1(:,j) =  tgrad1(:,j) - s*ksi2(:,u);
%            end
%         end
        if su == 0
            continue;
        end
        loss = loss + uloss/su;
        grad_ksi2 = grad_ksi2 + tgrad2/su;
        grad_ksi1 = grad_ksi1 + tgrad1/su;
    end
    J1 = loss/UC;
    object = J1;
   
     
    %% 重构误差项 第二、三项
    FM1 = sigmoid(W12*ksi1 + repmat(b12,1,IC)).*II;
    FM2 = sigmoid(W22*ksi2 + repmat(b22,1,UC)).*II';
    J2 = alpha * sum(sum((FM1 - Train_FM).*(FM1 - Train_FM)));
    J3 = alpha * sum(sum((FM2 - Train_FMT).*(FM2 - Train_FMT)));
    object = object + J2 + J3;
    
    CC = (FM1 - Train_FM).*FM1.*(1 - FM1);
    DD = (FM2 - Train_FMT).*FM2.*(1 - FM2);
    grad_ksi2 = grad_ksi2/UC + 2*alpha*W22'*DD;
    grad_ksi1 = grad_ksi1/UC + 2*alpha*W12'*CC;
    %% 正则化项 第四项
    J4 = gama * (sum(sum(W11.*W11)) + sum(sum(b11.*b11)) + sum(sum(W12.*W12)) + sum(sum(b12.*b12)) + sum(sum(W21.*W21)) + sum(sum(b21.*b21)) + sum(sum(W22.*W22)) + sum(sum(b22.*b22)));
    object = object + J4;

    if isTest == 0 
        [ndcg10, usercount10] = NDCGat10(Train_FM, All_FM, M);
        fprintf('NDCGat10 is %f\n', ndcg10);
        fprintf('usercount10 is %f\n', usercount10);
        fprintf(fid,'%f\t%d\t%f\t%f\t%f\t%f\t%f\n', ndcg10, usercount10, J1, J2, J3, J4, object);
    end

    AA = ksi1 .* (1 - ksi1);
    BB = ksi2 .* (1 - ksi2);
    %% 梯度
    grad_W11 = (grad_ksi1 .* AA)*Train_FMT + 2*gama*W11;
    grad_b11 = (grad_ksi1 .* AA)*ones(IC,1) + 2*gama*b11;  
    grad_W12 = 2*alpha*CC*ksi1' + 2*gama*W12;
    grad_b12 = 2*alpha*CC*ones(IC,1) + 2*gama*b12;
    
    grad_W21 = (grad_ksi2 .* BB)*Train_FM + 2*gama*W21;
    grad_b21 = (grad_ksi2 .* BB)*ones(UC,1) + 2*gama*b21;
    grad_W22 = 2*alpha*DD*ksi2' + 2*gama*W22;
    grad_b22 = 2*alpha*DD*ones(UC,1) + 2*gama*b22;
    
    grad = [grad_W11(:); grad_b11(:); grad_W12(:); grad_b12(:); grad_W21(:); grad_b21(:); grad_W22(:); grad_b22(:)];

end

% sigmoid
function sigm = sigmoid(x)
    sigm = 1./(1+ exp(-x));
end
% logistic loss ---- smooth
function p = getPenalty(p1,p2,q1,q2)
    d1 = p1 - p2;
    d2 = q1 - q2;
    if d1 > 0
        p = d1*log(1 + exp(-d2));
    elseif d1 < 0
        p = -d1*log(1 + exp(d2));
    else 
        p = 0;
    end
end
% 损失函数梯度
function g = getPenaltyGrad(p1,p2,q1,q2)
    d1 = p1 - p2;
    d2 = q1 - q2;
    if d1 > 0
        g = - d1 * exp(-d2)/(1+exp(-d2));
    elseif d1 < 0
        g = - d1 * exp(d2)/(1+exp(d2));
    else
        g = 0;
    end
end