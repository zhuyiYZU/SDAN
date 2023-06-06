function weight = initialize_au(k, Train_FM)
    % init the paramters of AutoEncoder via sae 
    % 商品隐特征权重学习
    [m, n] = size(Train_FM);
    theta = initialize_img1(k, m, Train_FM);  % 2*k*n+k+m
    W11 = reshape(theta(1:k*m),k,m);
    b11 = theta(k*m+1:k*m+k);
    W12 = reshape(theta(k*m+k+1:2*k*m+k),m,k);
    b12 = theta(2*k*m+k+1:2*k*m+m+k);
    % 用户隐特征权重学习
    Train_FM = Train_FM';
    theta = initialize_img1(k, n, Train_FM);  % 2*k*n+k+n
    W21 = reshape(theta(1:k*n),k,n);
    b21 = theta(k*n+1:k*n+k);
    W22 = reshape(theta(k*n+k+1:2*k*n+k),n,k);
    b22 = theta(2*k*n+k+1:2*k*n+k+n);
    weight = [W11(:);b11(:);W12(:);b12(:);W21(:);b21(:);W22(:);b22(:)];
    
%     filename = [indir '/Au_Weight'];
%     command = ['save ' filename ' W11 b11 W12 b12 W21 b21 W22 b22'];
%     eval(command);
end