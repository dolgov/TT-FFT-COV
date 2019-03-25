% Generates a random field from a product Gaussian vector realisation using 
% circulant embedding of the covariance matrix and TT approximations

model_Y     = 'matern';       % type of covariance function
variance_Y  = 10;                % variance parameter for covariance function
lambda_Y    = [10 20];           % vector of correlation length scales in y,x,z-directino
% lambda_Y    = [10 20 10];           % vector of correlation length scales in y,x,z-directino
lambda_Y = [1, 5, 10, 15, 20, 25, 30, 35];
micro_Y     = 1;                % fine-scale smoothing parameter
beta_Y      = 1;               % expected value for mean value
Qbb_Y       = 0.25;             % variance of mean value
domain_len  = [100 100];        % vector of domain length in y,x,z-direction
domain_len = [10, 50, 100, 150, 200, 250, 300, 350];
n_el        = [300 300];      % vector of element numbers in y,x,z-direction
n_el = [100, 100, 100, 100, 100, 100, 100, 100];
% domain_len  = [10 10 10];        % vector of domain length in y,x,z-direction
% n_el        = [30 30 30];      % vector of element numbers in y,x,z-direction
nel         = prod(n_el);       % total number of elements
el_len      = domain_len./n_el; % vector of element length in y,x,z-direction
flag_kit    = 0;                % spectrum allowed to be random
flag_zh     = 0;                % low-value inclusions, high-value conncetivity
periodicity = [0 0 0];          % non-periodic random field
tt_tol = 1e-4;

tic
Y = test_generate_y_tt(model_Y,variance_Y,lambda_Y,micro_Y,beta_Y,Qbb_Y,domain_len,el_len,n_el,nel,flag_kit,flag_zh,periodicity, tt_tol);
toc

return;

Y = full(Y);
surf(reshape(Y,n_el)), daspect([1 1 0.05])
% meanY = dot(tt_ones(Y.n),Y)/prod(Y.n);
meanY = mean(Y(:));
zlim([meanY-3*sqrt(variance_Y) meanY+3*sqrt(variance_Y)])
shading flat, lighting flat, material shiny, camlight head
colorbar
