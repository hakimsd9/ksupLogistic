function [w,costs] = ksupMulticlassLogistic(X,Y,lambda,k,w0, ...
                                        iters_acc,eps_acc);
% Authors: Matthew Blaschko - matthew.blaschko@inria.fr
%          Hakim Sidahmed - hakimsd@gmx.com
% Copyright (c) 2012-2013
%
% Run Ksupport norm using logistic loss function
% first 3 arguments are required!
%
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
%
% If you use this software in your research, please cite:
%
% M. B. Blaschko, A Note on k-support Norm Regularized Risk Minimization.
% arXiv:1303.6390, 2013.
%
% Argyriou, A., Foygel, R., Srebro, N.: Sparse prediction with the k-support
% norm. NIPS. pp. 1466-1474 (2012)

% k-support norm with logistic regression extended to a multiclass problem (iris dataset from UCI)

    if(~exist('eps_acc','var'))
        eps_acc = 1e-4;
    end

    if(~exist('iters_acc','var'))
        iters_acc = 2000; 
    end
    
    
    if(~exist('w0','var'))
      w0 = zeros((size(X,2)+1)*size(Y,2),1);
    end
    
    if(~exist('k','var'))
        k = round(size(X,2)/4);
    end

    numClasses = size(Y,2);
           
    if(size(X,1)>size(X,2)) % lipschitz constant for gradient
        e = ones(size(X,1),1);
        L = eigs([length(e) e'*X;X'*e X'*X],1)/4;
    else
        L = eigs([ones(size(X,1)) + X*X'],1)/4;
    end
    L = L*numClasses;

    [w,costs] = overlap_nest(@(w)(LogisticLoss(w,X,Y)),...
                             @(w)(gradLogisticLoss(w,X,Y)), lambda, ...
                             L, w0, k, iters_acc,eps_acc);
end

function l = LogisticLoss(w,X,Y)

  w = reshape(w,[size(X,2)+1 size(Y,2)]);
  P = exp(ones(size(X,1),1)*w(1,:) + X*w(2:end,:));
  P = P./repmat(sum(P,2),[1 size(P,2)]);
  l = -sum(sum(Y.*log(P) + (1-Y).*log(1-P)));
  l = l./size(X,1);

end

function g = gradLogisticLoss(w,X,Y)
 
w = reshape(w,[size(X,2)+1 size(Y,2)]);

e = ones(size(X,1),1);
  P = exp(ones(size(X,1),1)*w(1,:) + X*w(2:end,:));
  P = P./repmat(sum(P,2),[1 size(P,2)]);
  dYP = (Y-P)';
  g = -([dYP*e dYP*X])';
  g = g(:);

  g = g./size(X,1);
  
end

% end of file
