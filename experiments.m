function [accTab,mseTab,algName] = experiments();
% Authors: Matthew Blaschko - matthew.blaschko@inria.fr
%          Hakim Sidahmed  - hakimsd@gmx.com
% Copyright (c) 2013
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
%
%
% If you use this software in your research, please cite:
%
% M. B. Blaschko, A Note on k-support Norm Regularized Risk Minimization.
% arXiv:1303.6390, 2013.
%
% Argyriou, A., Foygel, R., Srebro, N.: Sparse prediction with the k-support
% norm. NIPS. pp. 1466-1474 (2012)


% k-support norm with logistic regression extended to a multiclass problem (iris dataset from UCI)

blaschkoDisp('iris data');
[X Y] = genDataIris('irisData.csv');
Xtrain = X(1:35,:);
Ytrain = Y(1:35,:);
Xval = X(36:70,:);
Yval = Y(36:70,:);
Xtest = X(70:end,:);
Ytest = Y(70:end,:);


d = size(Xtest,2);
% set of k values to select from for k-support norm
ks = [1:d];
numKsteps = 10;
% ks = [1:max(1,round(d/numKsteps)):d];
% set of regularization parameters to select from

lambdas = 10.^[-15:5];
%lambdas = 10.^[-15];

alg = cell(0);
algName = cell(0);
%alg{end+1} = @matlabLogistic; % just to debug and compare to a known
%                              % correct implementation of Logistic regression
%algName{end+1} = 'logistic loss matlab';
alg{end+1} = @ksupMulticlassLogistic; 
algName{end+1} = 'logistic loss';

accTab = zeros(3,length(alg));
mseTab = zeros(3,length(alg));


% squared loss
for i=1:length(algName)
    blaschkoDisp(['evaluating ' algName{i}])
    blaschkoDisp('k-support validation');
    [acc_k,mse_k] = evalMethod(alg{i},Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,lambdas,ks);
    blaschkoDisp('l1 regularization');
    [acc_1,mse_1] = evalMethod(alg{i},Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,lambdas,1);
    blaschkoDisp('l2 regularization');
    [acc_2,mse_2] = evalMethod(alg{i},Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,lambdas,d);
    accTab(:,i) = [acc_k; acc_1; acc_2];
    mseTab(:,i) = [mse_k; mse_1; mse_2];
end
end

function blaschkoDisp(message)
disp([message ' ' datestr(now)]);
end

function [acc,mse,beta] = evalMethod(func,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,lambdas,ks)

beta = modelSelection(func,Xtrain,Ytrain,Xval,Yval,lambdas,ks);

e = ones(size(Xtest,1),1);
pred = [e Xtest]*beta;
if(size(Xval,2)>1)
    [score,predind] = max(pred,[],2);
    [score,Ytestind] = max(Ytest,[],2);
    acc = length(find(predind==Ytestind))/length(Ytestind);
else
    acc = length(find(sign(pred)==Ytest))/length(Ytest);
end
mse = norm(Ytest-pred).^2;

end

function beta = modelSelection(func,Xtrain,Ytrain,Xval,Yval,lambdas,ks)

e = ones(size(Xval,1),1);
accval = -Inf;
    beta = zeros(size(Xval,2)*size(Yval,2),1); 
    for i=1:length(lambdas)
        for j = 1:length(ks)
            w = func(Xtrain,Ytrain,lambdas(i),ks(j));
            w = reshape(w,[size(Xtrain,2)+1 size(Ytrain,2)]);
            pred = [e Xval]*w;
            if(size(Xval,2)>1)
                [score,predind] = max(pred,[],2);
                [score,Yvalind] = max(Yval,[],2);
                acc = length(find(predind==Yvalind))/length(Yvalind);
            else
                acc = length(find(sign(pred)==Yval))/length(Yval);
            end
            if(acc>accval)
                accval = acc;
                beta = w;
            end
        end
    end
end

function [X Y] = genDataIris(filename)

numberClasses = 3;

fid = fopen(filename);
X = textscan(fid, '%f %f %f %f %*s', 'Delimiter', ',', 'CollectOutput', 1);
fclose(fid);
fid = fopen(filename);
Z = textscan(fid, '%*f %*f %*f %*f %s', 'Delimiter', ',', 'CollectOutput', 1);
fclose(fid);
X = X{1};
Z = Z{1};

Y = zeros(size(X,1), numberClasses);

for k = 1:size(X,1)
    if  strcmp(Z(k),'Iris-setosa')
        Y(k,:) = [1 0 0];
    elseif strcmp(Z(k),'Iris-versicolor') 
        Y(k,:) = [0 1 0];
    elseif strcmp(Z(k),'Iris-virginica')
        Y(k,:) = [0 0 1];
    end
end

% Shuffle the data
r = randperm(size(Y,1));
X = X(r,:);
Y = Y(r,:);

end
