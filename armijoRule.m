function betab = armijoRule(betab, diff_betab, K, y, para)
% armijoRule       armijo type rule 
% 
% Description
%   BETAB = ARMIJORULE(BETAB, DIFF_BETAB, K, Y, PARA) means doing a line
%   search step along the descent direction with the armijo type rule.
% 
% Input,
%   BETAB : the coefficient vector of the kernel function and the bias term ((N+1) x 1)
%   DIFF_BETAB : the difference value of two adjacent betab ((N+1) x 1)
%   K : indefinite kernel matrix (N x N)
%   Y : numerical label vector corresponding to the training samples in K above (N x 1)
%   PARA : model parameters of the IKSVM-DC model.
% 
% Output,
%   BETAB : new coefficient vector of the kernel function and the bias term ((N+1) x 1)
% 
% Extended description of input variables
%   PARA,
%       PARA.GAMMA : the regularization parameter
%       PARA.DELTA : tolerance during the iteration
%       PARA.MAX_ITER : maximum number of iterations 
% 
% Copyright: Hai-Ming Xu1 (heimingx@seu.edu.cn), Hui Xue1 (hxue@seu.edu.cn),
%   Xiao-Hong Chen2 (lyandcxh@nuaa.edu.cn), Yun-Yun Wang3 (wangyunyun@njupt.edu.cn)
%   1School of Computer Science and Engineering, Southeast University, Nanjing 210096, P.R.China
%   2College of Science, Nanjing University of Aeronautics and Astronautics, Nanjing, 210016, China
%   3School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, 210046, China
% 

diff_beta_norm_square = norm(diff_betab).^2;

upsilon = 3;
armijo_miu = 0.4;
armijo_eta = 0.5;

betab_obj = getObjectiveFunctionValue(betab, K, y, para);
while 1 
    betab_new = betab + upsilon * diff_betab;
    betab_new_obj = getObjectiveFunctionValue(betab_new, K, y, para);
    if betab_new_obj <= betab_obj - armijo_miu * upsilon * diff_beta_norm_square
        break;
    else
        upsilon = armijo_eta * upsilon;
    end
end
betab = betab_new;

end