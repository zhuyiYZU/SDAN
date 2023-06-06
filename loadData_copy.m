% RatingMatrix_Cell  CM  ѵ�����ݵİ�Ԫ����
% RatingMatrix_Full  FM  ѵ�����ݵ�ȫ����
% RatingMatrix_ACell ACM ѵ�����ݺ�Ԥ�����ݽ����İ�Ԫ����
% ///RatingMatrix_AFull AFM ѵ�����ݺ�Ԥ�����ݽ�����ȫ����
function [Train4train_CM, Train4train_FM, Train_CM, Train_FM, All_FM] = loadData()
%     global dataDir;
%     global arffFile;
%     filename = [dataDir '/' arffFile '.train.4train.CM'];
%     fprintf('%s\n',filename);
    filename = 'C:\Users\pc\Documents\MATLAB\REAP\data\ml-100k\u1.base';
    M = load(filename);
    % ��ÿ���û�����Ʒ��Ϣ�洢��һ��cellԪ����
    % ÿ��Ԫ������һ��2�еľ��󣬵�һ��Ϊ��Ʒ���������ڶ���Ϊ��Ʒ������ֵ
    Train4train_CM = col2tocellmatrix(M);
    filename = [dataDir '/Train4train_CM'];
    command = ['save ' filename ' Train4train_CM'];
    eval(command);
    
    % ��ѵ�������У��û���Ʒ����洢Ϊ������
    filename = [dataDir '/' arffFile '.train.4train.FM'];
    fprintf('%s\n',filename);
    Train4train_FM = load(filename);
    filename = [dataDir '/Train4train_FM'];
    command = ['save ' filename ' Train4train_FM'];
    eval(command);
    
%     % ����֤�����У��û���Ʒ����洢Ϊ��Ԫ����
%     filename = [dataDir '/' arffFile '.train.4valid.CM'];
%     fprintf('%s\n',filename);
%     M = load(filename);
%     Train4valid_CM = col2tocellmatrix(M);
%     filename = [dataDir '/Train4valid_CM'];
%     command = ['save ' filename ' Train4valid_CM'];
%     eval(command);
    
    % ������ѵ�����ݴ洢Ϊ��Ԫ����
    filename = [dataDir '/' arffFile '.train.CM'];
    fprintf('%s\n',filename);
    M = load(filename);
    Train_CM = col2tocellmatrix(M);
    filename = [dataDir '/Train_CM'];
    command = ['save ' filename ' Train_CM'];
    eval(command);
    
    
    % ������ѵ�����ݴ洢Ϊ������
    filename = [dataDir '/' arffFile '.train.FM'];
    fprintf('%s\n',filename);
    Train_FM = load(filename);
    filename = [dataDir '/Train_FM'];
    command = ['save ' filename ' Train_FM'];
    eval(command);
    
%     % ���������ݴ洢Ϊ��Ԫ����
%     filename = [dataDir '/' arffFile '.test.CM'];
%     fprintf('%s\n',filename);
%     M = load(filename);
%     Test_CM = col2tocellmatrix(M);
%     filename = [dataDir '/Test_CM'];
%     command = ['save ' filename ' Test_CM'];
%     eval(command);
%     
%     % ���������ݣ��洢Ϊ��Ԫ����
%     filename = [dataDir '/' arffFile '.all.CM'];
%     fprintf('%s\n',filename);
%     M = load(filename);
%     All_CM = col2tocellmatrix(M);
%     filename = [dataDir '/All_CM'];
%     command = ['save ' filename ' All_CM'];
%     eval(command);
    
    % �����������У��û���Ʒ����洢Ϊ������
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