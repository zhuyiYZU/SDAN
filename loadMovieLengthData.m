% RatingMatrix_Cell  CM  训练数据的胞元矩阵
% RatingMatrix_Full  FM  训练数据的全矩阵
% RatingMatrix_ACell ACM 训练数据和预测数据交集的胞元矩阵
% ///RatingMatrix_AFull AFM 训练数据和预测数据交集的全矩阵
function [Train_CM, Train_FM, All_FM,Q] = loadMovieLengthData(itemNum,userNum)
 
    % 将所有训练数据存储为满矩阵
     % Train_FM中，m个user，n个item，M是n*m 
     filename = 'C:\Users\Lenovo\Documents\MATLAB\REAP\data\ml-100k\u1.base';
     originalData = load(filename);
     Train_FM = saveFM(userNum,itemNum,originalData);
     
     % 生产Q
     Q = generateQByFM(Train_FM);
    
     % 将所有训练数据存储为胞元矩阵
     % 将每个用户的商品信息存储到一个cell元素中
     % 每个元素中是一个2行的矩阵，第一行为商品的索引，第二行为商品的评分值
     Train_CM = col2tocellmatrix(Train_FM);
    
    % 将所有数据中，用户商品矩阵存储为满矩阵
     filename = 'C:\Users\Lenovo\Documents\MATLAB\REAP\data\ml-100k\u.data';
     allData = load(filename);
     All_FM = saveFM(userNum,itemNum,allData);
end

function FM = saveFM(userNum,itemNum,data)
    FM = zeros(userNum,itemNum);
    for rowNum=1:size(data,1)
        userIndex = data(rowNum,1);
        itemIndex = data(rowNum,2);
        FM(userIndex,itemIndex) = data(rowNum,3);
    end
end

function Q = generateQByFM(FM)
    Q = zeros(size(FM,1),size(FM,2));
    for rowNum=1:size(FM,1)
        for colNum=1:size(FM,2)
            if(FM(rowNum,colNum) == 0)
                Q(rowNum,colNum) = 0;
            else
                Q(rowNum,colNum) = 1;
            end
        end
    end
end

function CM =  col2tocellmatrix(FM)
    CM = {};
    for rowNum=1:size(FM,1)
         info = [];
         count = 0;
        
        for colNum=1:size(FM,2)
            if(FM(rowNum,colNum) ~= 0)
                count = count+1;
                info(1,count) = colNum;
                info(2,count) = FM(rowNum,colNum);
            end
        end
        
        CM{rowNum,1} = info;
    end
end