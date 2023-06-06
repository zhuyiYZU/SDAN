function control4validation()
    global dataDir;
    global resultDir;
    global arffFile;
    global UC;  % userCount
    global IC;  % itemCount

    arffFile = 'movieLens_100K';
    %arffFile = 'douban_book';
    %arffFile = 'yelp_hin_selected';
    %arffFile = 'netflix_3m1k';
    %arffFile = 'movieLens_1M';
    ratio = [5 10 15 20 25];
    for i = 1:5
       dataDir = [arffFile '_N+15/data/' num2str(ratio(1,i))];
       resultDir = [arffFile '_N+15/param-result/' num2str(ratio(1,i))];
       if ~exist(resultDir)
           mkdir(resultDir);
       end
       [Train4train_CM, Train4train_FM, Train_CM, Train_FM, All_FM] = loadData();
       [UC, IC] = size(Train_FM);     
       %%%%%%%% Parameters Tuning %%%%%%%
       rank = [1 5 10 15 20];
       for j = 4:4
           for k = 1:1
               for z = 1:1
                   %%%%%%%%%%%% set parameters
                   cur_rank = rank(1,j);
                   cur_alpha = 10;
                   cur_gama = 1;
                   cur_iter = 100;
                   isTest = 0;
                   
                   fprintf('\n\n======================================================\n')
                   fprintf('runing params now : %s_0.%d\t%d\t%f\t%f\n', arffFile, ratio(1,i), cur_rank, cur_alpha, cur_gama);
                   
                   %%%%%%%%%%%% train begin
                   weight = initialize_au(cur_rank, Train4train_FM);
                   RSVD(Train4train_CM, Train4train_FM, Train_FM, weight, cur_rank, cur_alpha, cur_gama, cur_iter, isTest); 
               end
           end
       end
       %%%%%%% End of Parameters Tuning %%%%%%%%%
    end
end