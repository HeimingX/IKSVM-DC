function dc = DCdecomposition(K)
% DCdecomposition       Construct DC decomposition for the objective function.
% 
% Description
%   DC = DCDECOMPOSITION(K) means DC decomposition of objective function
% 
% Input,
%   K : indefinite kernel matrix (N x N)
% 
% Output,
%   DC : DC decomposition terms
% 
% Extended description of ouput variables
% DC,
%   DC.DC_G_BETA : the first term of DC decomposition in (12) in our paper
%   DC.DC_H_BETA : the second term of DC decomposition in (12) in our paper.
% 
% 
% Copyright: Hai-Ming Xu1 (heimingx@seu.edu.cn), Hui Xue1 (hxue@seu.edu.cn),
%   Xiao-Hong Chen2 (lyandcxh@nuaa.edu.cn), Yun-Yun Wang3 (wangyunyun@njupt.edu.cn)
%   1School of Computer Science and Engineering, Southeast University, Nanjing 210096, P.R.China
%   2College of Science, Nanjing University of Aeronautics and Astronautics, Nanjing, 210016, China
%   3School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, 210046, China
% 

row = size(K, 1);
[~,eigVal] = eig(K);
eigVal = real(eigVal);
max_eigVal = max(max(eigVal));
rho_max = max_eigVal;

salt = 1e-4;
% decomposition:K = rho*I - (rho*I-K) 
rho = rho_max+salt;
dc_g_beta = rho.*eye(row);
dc_h_beta = dc_g_beta - K;

dc.dc_g_beta = dc_g_beta;
dc.dc_h_beta = dc_h_beta;

end