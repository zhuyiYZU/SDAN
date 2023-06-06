clear all;
addpath(genpath('C:\Users\Lenovo\Documents\MATLAB\REAP\DeepLearnToolbox'));
addpath(genpath('C:\Users\Lenovo\Documents\MATLAB\REAP\fastfit'));
addpath(genpath('C:\Users\Lenovo\Documents\MATLAB\REAP\lightspeed'));
addpath(genpath('C:\Users\Lenovo\Documents\MATLAB\REAP\logreg'));
addpath(genpath('C:\Users\Lenovo\Documents\MATLAB\REAP\minFunc'));

alpha = 500;
gamma = 0.00001;
numK = 45;  % number of hidden units

for times=1:6
    
    all_result = zeros(5,6);
    
    [Train_CM, Train_FM, All_FM, Q] = loadImbalancedMovieLengthData(1682, 943);
 
    %%Initialize the parameter
    theta = initialize_au(numK, Train_FM);    % Randomly initialize the parameters
    [UC, IC] = size(Train_FM);
    index = 0;
    W11 = reshape(theta(index+1:index+numK*UC),numK,UC);
    b11 = theta(index+numK*UC+1:index+numK*UC+numK);
    W12 = reshape(theta(index+numK*UC+numK+1:index+2*numK*UC+numK),UC,numK);
    b12 = theta(index+2*numK*UC+numK+1:index+2*numK*UC+numK+UC);
    index = index + 2*numK*UC + numK + UC;
    W21 = reshape(theta(index+1:index+numK*IC),numK,IC);
    b21 = theta(index+numK*IC+1:index+numK*IC+numK);
    W22 = reshape(theta(index+numK*IC+numK+1:index+2*numK*IC+numK),IC,numK);
    b22 = theta(index+2*numK*IC+numK+1:index+2*numK*IC+numK+IC);
    
    %% Use minFunc to minimize the function
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost function.
    options.maxIter = 300;	  % Maximum number of iterations of L-BFGS to run
    options.display = 'on';
    options.TolFun  = 1e-6;
    options.TolX = 1e-1119;
    %       options.TolX = 0;
    options.maxFunEvals = 300;
    %       options.maxFunEvals = 100000;
    [optweight, cost] = minFunc( @(p) computeObjAndGradMine(p, numK, alpha, gamma, Q, Train_FM, Train_CM, All_FM, options.maxFunEvals, 1), theta, options);
    fprintf('cost is %f\n', cost); 
    
    %%%%%%%%%%%%
    W11 = reshape(optweight(1:numK*UC),numK,UC);
    b11 = optweight(numK*UC+1:numK*UC+numK);
    index = 2*numK*UC+numK+UC;
    W21 = reshape(optweight(index+1:index+numK*IC),numK,IC);
    b21 = optweight(index+numK*IC+1:index+numK*IC+numK);
    ksi1 = sigmoid(W11*Train_FM + repmat(b11,1,IC));
    ksi2 = sigmoid(W21*Train_FM'+ repmat(b21,1,UC));
    M = ksi2'*ksi1;
    precision = AvrgPrecision(Train_FM, All_FM, M);
    fprintf('AvrgPrecision is %f\n', precision);
%     fprintf(fid,'%f\t%f\t%f\t%f\t%d\t%d\n', ndcg15, ndcg10, ndcg5, precision, usercount10, uc_avgscore);
%     fclose(fid);
end;