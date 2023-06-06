% RatingMatrix_Cell  CM  ѵ�����ݵİ�Ԫ����
% RatingMatrix_Full  FM  ѵ�����ݵ�ȫ����
% RatingMatrix_ACell ACM ѵ�����ݺ�Ԥ�����ݽ����İ�Ԫ����
% ///RatingMatrix_AFull AFM ѵ�����ݺ�Ԥ�����ݽ�����ȫ����
function [Train_CM, Train_FM, All_FM,Q] = loadMovieLengthData(itemNum,userNum)
 
    % ������ѵ�����ݴ洢Ϊ������
     % Train_FM�У�m��user��n��item��M��n*m 
     filename = 'C:\Users\Lenovo\Documents\MATLAB\REAP\data\ml-100k\u1.base';
     originalData = load(filename);
     Train_FM = saveFM(userNum,itemNum,originalData);
     
     % ����Q
     Q = generateQByFM(Train_FM);
    
     % ������ѵ�����ݴ洢Ϊ��Ԫ����
     % ��ÿ���û�����Ʒ��Ϣ�洢��һ��cellԪ����
     % ÿ��Ԫ������һ��2�еľ��󣬵�һ��Ϊ��Ʒ���������ڶ���Ϊ��Ʒ������ֵ
     Train_CM = col2tocellmatrix(Train_FM);
    
    % �����������У��û���Ʒ����洢Ϊ������
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