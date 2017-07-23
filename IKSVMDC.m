function model = IKSVMDC(K, y, para)
% IKSVM-DC      Indefinite Kernel SVM with Difference of Convex Functions Programming
% 
% Description
%   MODEL = IKSVMDC(K, Y, PARA) means Indefinite Kernel SVM with DCP
% 
%   Statement
%   The function model = IKSVMDC(K, y, para) here is specially designed 
%   based on difference of convex functions programming proposed by Dinh 
%   and Le Thi et al. We adapt the classical DC algorithm for the primal 
%   form of indefinite kernel SVM problem.
% 
%   Inputs,
%       K : indefinite kernel matrix (N x N)
%       Y : numerical label vector corresponding to the training samples in K above (N x 1)
%       PARA : model parameters of the IKSVM-DC model.
%       
%   Outputs,
%       MODEL : training model parameters
% 
%   Extended description of input/ouput variables
%   PARA,
%       PARA.GAMMA : the regularization parameter
%       PARA.DELTA : tolerance during the iteration
%       PARA.MAX_ITER : maximum number of iterations
%   MODEL,
%       MODEL.BETA : the coefficient vector of the kernel function (N x 1)
%       MODEL.B : the bias term
%       MODEL.ITER ; the number of iteration
% 
% 
% Copyright: Hai-Ming Xu1 (heimingx@seu.edu.cn), Hui Xue1 (hxue@seu.edu.cn),
%   Xiao-Hong Chen2 (lyandcxh@nuaa.edu.cn), Yun-Yun Wang3 (wangyunyun@njupt.edu.cn)
%   1School of Computer Science and Engineering, Southeast University, Nanjing 210096, P.R.China
%   2College of Science, Nanjing University of Aeronautics and Astronautics, Nanjing, 210016, China
%   3School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, 210046, China
% 

fprintf(1,'IKSVM-DC is training...\n');

% make sure K is symmetric
K = (K + K')/2;
row = size(K, 1);

% decomposition of the non-convex term
dc = DCdecomposition(K);

% initialization
rng(123);
beta_t = 2 * rand(row, 1) - 1;
iter = 0;
b_t = 1;

g_beta = 1/2 .* para.gamma .* dc.dc_g_beta;

warning off all;
while 1
    iter = iter + 1;
    
    % solve theta
    theta_t_new = para.gamma .* beta_t' * dc.dc_h_beta;
    
    % solve beta
    cvx_begin quiet
        cvx_solver Mosek;
        variable beta_new(row);
        variable b;
        
        minimize( quad_form(beta_new,g_beta)+1/2.*sum_square_pos(ones(row,1)-y.*(K*beta_new+b.*ones(row,1)))-theta_t_new*beta_new);
    cvx_end
    if any(isnan(beta_new))
        warning('cvx throw out a NaN error when solve beta') %#ok<WNTAG>
        break;
    end
    beta_t_new = beta_new;
    b_t_new = b;
    
    % do a line search along the descent direction to accelerate the convergence rate
    betab = [beta_t; b_t];
    betab_new = [beta_t_new; b_t_new];
    diff_betab = betab_new - betab;
    betab_new = armijoRule([beta_t_new; b_t_new], diff_betab, K, y, para);
    beta_t_new = betab_new(1 : end-1);
    b_t_new = betab_new(end);
    
    % stop criterion
    % d_beta not greater than para.delta
    d_beta = norm(beta_t_new - beta_t)^2;
    if d_beta <= para.delta
        beta_t = beta_t_new;
        b_t = b_t_new;
        break;
    end
    
    % objective function value not less than 0
    obj_value = getObjectiveFunctionValue([beta_t_new; b_t_new], K, y, para);
    if obj_value < 0
        break;
    end
    
    % upper bound of iteration 
    if iter > para.max_iter
        break;
    end
    
    % update solutions
    beta_t = beta_t_new;
    b_t = b_t_new;
end

model.beta = beta_t;
model.b = b_t;
model.iter = iter;

end