function accuracy = predict(test_K, test_y, model)
% predict       Calculate the predicted accuracy for test_K.
% 
% Description
%   ACCURACY = PREDICT(TEST_K, TEST_Y, MODEL) calculates the predicted 
%       accuracy for test_K.
% 
% Input,
%   TEST_K : indefinite kernel matrix for test set(N x M)
%   TEST_Y : numerical label vector corresponding to the testing samples in TEST_K above (M x 1)
%   MODEL : training model parameters.
% 
% Output,
%   ACCURACY : the accuracy of predicting with the MODEL.
% 
% Extended description of input variables
%   MODEL,
%       MODEL.BETA : the coefficient vector of the kernel function (N x 1)
%       MODEL.B : the bias term
%       MODEL.ITER ; the number of iteration
% 
% Copyright: Hai-Ming Xu1 (heimingx@seu.edu.cn), Hui Xue1 (hxue@seu.edu.cn),
%   Xiao-Hong Chen2 (lyandcxh@nuaa.edu.cn), Yun-Yun Wang3 (wangyunyun@njupt.edu.cn)
%   1School of Computer Science and Engineering, Southeast University, Nanjing 210096, P.R.China
%   2College of Science, Nanjing University of Aeronautics and Astronautics, Nanjing, 210016, China
%   3School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, 210046, China
% 

fprintf(1,'\nPredict for the test data.\n');

predict_y = model.beta' * test_K + model.b;
accuracy = mean(sign(predict_y') == test_y);

end