function run_BMR(arffFile)
    %arffFile = 'movieLens_100K';
    %arffFile = 'douban_book';
    %arffFile = 'yelp_hin_selected';
    %arffFile = 'netflix_3m1k';
    %arffFile = 'movieLens_1M';
    ratio = [5 10 15 20 25];
    
    for times=1:3
        all_result = zeros(5,6);
        for i = 1:5
            dataDir = [arffFile '_N+15/data/' num2str(ratio(1,i))];
            filename = [dataDir '/' arffFile '.train.FM'];
            fprintf('%s\n',filename);
            Train_FM = load(filename);
           
            filename = [dataDir '/' arffFile '.all.FM'];
            fprintf('%s\n',filename);
            All_FM = load(filename);
            
            Test_FM = All_FM;
            Test_FM(find(Train_FM)) = 0;
           
            train = sparse(Train_FM);
            test = sparse(Test_FM);
            [M, N] = size(train);
            predict = BMR(M,N,train,test);
            predict_FM = full(predict);
            
            [ndcg15, ndcg10, ndcg5, usercount, scoreless15] = NDCGatK(Train_FM, All_FM, predict_FM);
            precision = AvrgPrecision(Train_FM, All_FM, predict_FM);
            
            all_result(1,i) = ndcg15;
            all_result(2,i) = ndcg10;
            all_result(3,i) = ndcg5;
            all_result(4,i) = precision;
            all_result(5,i) = M;
            all_result(6,i) = usercount;
            all_result(7,i) = scoreless15;
            fprintf('%s\n',ndcg15);
            fprintf('%s\n',ndcg10);
            fprintf('%s\n',ndcg5);
            fprintf('AvrgPrecision is %f\n', precision);
            fprintf('%d\n',usercount);
            fprintf('%d\n',scoreless15);
        end 
        result_file = [arffFile '_N+15/baseline/BMR_' num2str(times)];
        command = ['save ' result_file ' all_result'];
        eval(command);
    end

end