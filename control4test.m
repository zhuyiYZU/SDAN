function control4test()
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
    
    for times=1:5
       for i = 1:5
           dataDir = [arffFile '_N+15/data/' num2str(ratio(1,i))];
           resultDir = [arffFile '_N+15/result' num2str(times) '/' num2str(ratio(1,i))];
           if ~exist(resultDir)
               mkdir(resultDir);
           end
           [Train4train_CM, Train4train_FM, Train_CM, Train_FM, All_FM] = loadData();
           [UC, IC] = size(Train_FM);
           prank = 45;
           palpha = 500;
           pgama = 100; 
           isTest = 1;
           weight = initialize_au(prank, Train_FM);
           % ========== fixed 100 iteration ======
           fiter = 100;
           RSVD(Train_CM, Train_FM, All_FM, weight, prank, palpha, pgama, fiter, isTest); 
           % ========== fixed 150 iteration ======
           fiter = 150;
           RSVD(Train_CM, Train_FM, All_FM, weight, prank, palpha, pgama, fiter, isTest);  
        end 
    end
    
end