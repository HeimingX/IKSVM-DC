function obj_value = getObjectiveFunctionValue(betab, K, y, para)
% getObjectiveFunctionValue       get the value of objective function
% 
% Description
%   OBJ_VALUE = GETOBJECTIVEFUNCTIONVALUE(BETAB, K, Y, PARA) means get the
%       value of objective function with current beta.
% 
% Input,
%   BETAB : the coefficient vector of the kernel function and the bias term ((N+1) x 1)
%   K : indefinite kernel matrix (N x N)
%   Y : numerical label vector corresponding to the training samples in K above (N x 1)
%   PARA : model parameters of the IKSVM-DC model.
% 
% Output,
%   OBJ_VALUE : the value of objective function
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

beta = betab(1:end-1);
b = betab(end);
row = length(beta);

part1 = para.gamma * beta' * K * beta;
part2 = ones(1,row) * max(0, ones(row, 1) - y.*(K * beta + b .* ones(row, 1))).^2;
obj_value = 0.5 * (part1 + part2);

end