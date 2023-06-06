% isTest : to distinguish the process of validation and test.
% isTest == 0, it is validation data and tuning parameters now.
% isTest == 1, it is test data and check the parameters.
function RSVD(Train_CM, Train_FM, All_FM, weight, k, alpha, gama, iter, isTest)
    global resultDir;
    global UC;
    global IC;
    
    global params;
    global MinEpath;
    
    II = Train_FM;
    II(find(II>0)) = 1;
    
    params = [num2str(k), '_',  num2str(alpha), '_', num2str(gama)];
    MinEpath = [resultDir '/' params];
    iterPath =  [resultDir '/',params, '_', num2str(iter)];
    fid = fopen([iterPath, '_', num2str(isTest)],'a');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   addpath minFunc/
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost function. 
    options.maxIter = iter;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';
    options.TolFun  = 1e-6;
%   options.TolX = 1e-1111119;
    options.TolX = 0;
    options.maxFunEvals = 100000;
    [optweight, cost] = minFunc( @(p) computeObjAndGrad(p, k, alpha, gama, II, Train_FM, Train_CM, All_FM, fid, isTest), weight, options);      
    
    %%%%%%%%%%%%
    W11 = reshape(optweight(1:k*UC),k,UC);
    b11 = optweight(k*UC+1:k*UC+k);
    index = 2*k*UC+k+UC;
    W21 = reshape(optweight(index+1:index+k*IC),k,IC);
    b21 = optweight(index+k*IC+1:index+k*IC+k);  
    ksi1 = sigmoid(W11*Train_FM + repmat(b11,1,IC));
    ksi2 = sigmoid(W21*Train_FM'+ repmat(b21,1,UC));
    M = ksi2'*ksi1;
    % write the result of the last iteration to file
    if isTest == 1
        precision = AvrgPrecision(Train_FM, All_FM, M);
        fprintf('AvrgPrecision is %f\n', precision);
        [ndcg15, ndcg10, ndcg5, usercount10, uc_avgscore] = NDCGatK(Train_FM, All_FM, M);
        fprintf('NDCGat15 is %f\n', ndcg15);
        fprintf('NDCGat10 is %f\n', ndcg10);
        fprintf('NDCGat5 is %f\n', ndcg5);
        fprintf(fid,'%f\t%f\t%f\t%f\t%d\t%d\n', ndcg15, ndcg10, ndcg5, precision, usercount10, uc_avgscore);
        fclose(fid);
    end
    
%     if isTest == 1
%         A = M;
%         A(find(All_FM == 0)) = 0;
%         A(find(Train_FM > 0)) = 0;
%         B = All_FM;
%         B(find(Train_FM > 0)) = 0;
%         [userCount, itemCount] = size(A);
%         
%         scorefile = fopen([iterPath '_score'],'a');
%         fprintf(scorefile, '%d %d\n', userCount, itemCount);
%         for u = 1:userCount
%            uu = A(u, :);
%            [rowindex, columnindex, value] = find(uu);
%            fprintf(scorefile, '%d ', columnindex);
%            fprintf(scorefile, '\n');
%            fprintf(scorefile, '%f ', value);
%            fprintf(scorefile, '\n');
%         end
%         fclose(scorefile);
%         
%         relevancefile = fopen([iterPath '_relevance'],'a');
%         fprintf(relevancefile, '%d %d\n', userCount, itemCount);
%         for u = 1:userCount
%            uu = B(u, :);
%            [rowindex, columnindex, value] = find(uu);
%            fprintf(relevancefile, '%d ', columnindex);
%            fprintf(relevancefile, '\n');
%            fprintf(relevancefile, '%f ', value);
%            fprintf(relevancefile, '\n');
%         end
%         fclose(relevancefile);
%     end
end
% sigmoid
function sigm = sigmoid(x)
    sigm = 1./(1+ exp(-x));
end
