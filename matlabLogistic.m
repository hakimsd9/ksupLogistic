function [w,costs] = ksupMulticlassLogistic(X,Y,lambda,k,w0, ...
                                        iters_acc,eps_acc);
% Authors: Matthew Blaschko - matthew.blaschko@inria.fr
%          Hakim Sidahmed - hakimsd@gmx.com
% Copyright (c) 2012-2013
%
% Run logistic loss function using built in matlab
% first 2 arguments are required!
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

w = mnrfit(X,Y,'interactions','on');
costs = [];
w = [w zeros(size(w,1),1)];
return;

end
