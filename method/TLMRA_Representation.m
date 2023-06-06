function [ xFinalRepresentation ] = TLMRA_Representation( trainData,testData,trainLabels,parameters,maniparameters)

%% zca
xx = [trainData testData];
xZCAWhite = ZCA_Gen(xx);

%% tlda
xFinalRepresentation = TLDA(xZCAWhite,trainLabels,parameters);

%% manifold
% xFinalRepresentation = ManifoldRegression(xTldaRepresentation,maniparameters);

end

