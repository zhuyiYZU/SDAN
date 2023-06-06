function [L] = LaplacianMatrix(X,k)
%     options.p = 1024;             % keep default
    options.p = k;
    % options.sigma = 0.1;        % keep default
    % options.lambda = 10.0;      % keep default
    % options.gamma = 1.0;        % [0.1,10]
    % options.ker = 'rbf';        % 'rbf' | 'linear'
   
    % Data normalization
    X = X*diag(sparse(1./sqrt(sum(X.^2))));
   
    % Construct graph Laplacian
    manifold.k = options.p;
%     manifold.Metric = 'Cosine';
%     manifold.NeighborMode = 'KNN';
%     manifold.WeightMode = 'Cosine';
%    
%     W = graph(X',manifold);
%     Dw = diag(sparse(sqrt(1./sum(W))));
%     L = speye(nm)-Dw*W*Dw;
    manifold.Metric = 'Euclidean';
    manifold.NeighborMode = 'KNN';
    manifold.WeightMode = 'HeatKernel';
    

    S = graphNew(X',manifold);
    Ssum = sum(S,2);
    for iter=1:size(Ssum,1)
        D(iter,iter) = Ssum(iter);
    end
    L = D - S;
end