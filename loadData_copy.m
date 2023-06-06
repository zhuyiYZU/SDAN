% RatingMatrix_Cell  CM  训练数据的胞元矩阵
% RatingMatrix_Full  FM  训练数据的全矩阵
% RatingMatrix_ACell ACM 训练数据和预测数据交集的胞元矩阵
% ///RatingMatrix_AFull AFM 训练数据和预测数据交集的全矩阵
function [Train4train_CM, Train4train_FM, Train_CM, Train_FM, All_FM] = loadData()
%     global dataDir;
%     global arffFile;
%     filename = [dataDir '/' arffFile '.train.4train.CM'];
%     fprintf('%s\n',filename);
    filename = 'C:\Users\pc\Documents\MATLAB\REAP\data\ml-100k\u1.base';
    M = load(filename);
    % 将每个用户的商品信息存储到一个cell元素中
    % 每个元素中是一个2行的矩阵，第一行为商品的索引，第二行为商品的评分值
    Train4train_CM = col2tocellmatrix(M);
    filename = [dataDir '/Train4train_CM'];
    command = ['save ' filename ' Train4train_CM'];
    eval(command);
    
    % 将训练数据中，用户商品矩阵存储为满矩阵
    filename = [dataDir '/' arffFile '.train.4train.FM'];
    fprintf('%s\n',filename);
    Train4train_FM = load(filename);
    filename = [dataDir '/Train4train_FM'];
    command = ['save ' filename ' Train4train_FM'];
    eval(command);
    
%     % 将验证数据中，用户商品矩阵存储为胞元矩阵
%     filename = [dataDir '/' arffFile '.train.4valid.CM'];
%     fprintf('%s\n',filename);
%     M = load(filename);
%     Train4valid_CM = col2tocellmatrix(M);
%     filename = [dataDir '/Train4valid_CM'];
%     command = ['save ' filename ' Train4valid_CM'];
%     eval(command);
    
    % 将所有训练数据存储为胞元矩阵
    filename = [dataDir '/' arffFile '.train.CM'];
    fprintf('%s\n',filename);
    M = load(filename);
    Train_CM = col2tocellmatrix(M);
    filename = [dataDir '/Train_CM'];
    command = ['save ' filename ' Train_CM'];
    eval(command);
    
    
    % 将所有训练数据存储为满矩阵
    filename = [dataDir '/' arffFile '.train.FM'];
    fprintf('%s\n',filename);
    Train_FM = load(filename);
    filename = [dataDir '/Train_FM'];
    command = ['save ' filename ' Train_FM'];
    eval(command);
    
%     % 将测试数据存储为胞元矩阵
%     filename = [dataDir '/' arffFile '.test.CM'];
%     fprintf('%s\n',filename);
%     M = load(filename);
%     Test_CM = col2tocellmatrix(M);
%     filename = [dataDir '/Test_CM'];
%     command = ['save ' filename ' Test_CM'];
%     eval(command);
%     
%     % 将所有数据，存储为胞元矩阵
%     filename = [dataDir '/' arffFile '.all.CM'];
%     fprintf('%s\n',filename);
%     M = load(filename);
%     All_CM = col2tocellmatrix(M);
%     filename = [dataDir '/All_CM'];
%     command = ['save ' filename ' All_CM'];
%     eval(command);
    
    % 将所有数据中，用户商品矩阵存储为满矩阵
    filename = [dataDir '/' arffFile '.all.FM'];
    fprintf('%s\n',filename);
    All_FM = load(filename);
    filename = [dataDir '/All_FM'];
    command = ['save ' filename ' All_FM'];
    eval(command);
end
function CM =  col2tocellmatrix(M)
    CM = {};
    rownum = size(M, 1);
    i = 1;
    j = 1;
    while i<rownum
        itemnum = M(i,1);
        tmp = M(i+1:i+itemnum,:);
        CM{j,1} = tmp';
        i = i + 1 + itemnum;
        j = j + 1;
    end
end