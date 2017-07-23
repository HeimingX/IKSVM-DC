% DEMO  The example of IKSVM-DC algorithm
% 
%   Description
%   The algorithm IKSVM-DC is specially designed for indefinite kernel SVM
%   problem.
% 
% Copyright: Hai-Ming Xu1 (heimingx@seu.edu.cn), Hui Xue1 (hxue@seu.edu.cn),
%   Xiao-Hong Chen2 (lyandcxh@nuaa.edu.cn), Yun-Yun Wang3 (wangyunyun@njupt.edu.cn)
%   1School of Computer Science and Engineering, Southeast University, Nanjing 210096, P.R.China
%   2College of Science, Nanjing University of Aeronautics and Astronautics, Nanjing, 210016, China
%   3School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, 210046, China
% 

% load data file
load CoilYork

% initialize the model parameters.
para.gamma = 2^3; % the regularization parameter
para.delta = 1e-3; % tolerance during the iteration
para.max_iter = 300; % maximum number of iterations

% train primal form of indefinite kernel svm with IKSVM-DC
model = IKSVMDC(train_K, train_y, para);

% predict
accuracy = predict(test_K, test_y, model)