function Y = test_generate_y_tt(model_Y,variance_Y,lambda_Y,micro_Y,beta_Y,Qbb_Y,domain_len,el_len,n_el,nel,flag_kit,flag_zh,periodicity, tt_tol)
% TEST_GENERATE_Y generates realizations using spectral method
% (Dietrich & Newsam / Kitanidis). Uses additional microscale
% smoothing at boundaries (with lambda_Y) in order to do cheap periodic embedding.
% version 16 november 2006 / WN
% 
% All multivariate operations are performed using TT-Toolbox
%
% required input parameters:
% model_Y      : parametric geostatistical model
% variance_Y   : variance of model
% lambda_Y     : vector of correlation length in y,x,z-direction
% micro_Y      : microscale smoothing parameter relative to lambda_Y
%                (scalar, after Kitanidis book)
% beta_Y       : uncertain mean value
% Qbb_Y        : variance of uncertain mean value
% domain_len   : length of domain in y,x,z-direction
% el_len       : length of elements in y,x,z-direction
% n_el         : number of elements in y,x,z-direction
% nel          : number of elements (total)
%
% optional input paramters:
% flag_kit     : using kitanidis method to enforce spectrum.
%                0 - dietrich and newsam method
%                1 - kitanidis method
%                2 - kitanidis method with enforced gaussian mean value
% flag_zh:     : using zinn&harvey method to generate connectivity,
%                0 - standard fields
%                1 - low- K inclusions
%                2 - high-K inclusions
% periodicity  : vector of flags for periodicity along y,x,z-direction
%                [1 0 0] - periodic in y-direction
%                [0 1 0] - periodic in x-direction
%                [0 0 1] - periodic in z-direction
%                ...and all combinations of y,x,z as desired
% tt_tol       : tolerance for TT approximation [1e-6]

d = numel(n_el); % dimension

% checking optional input parameters
if nargin < 11 || isempty(flag_kit)
    flag_kit = false;
end
if nargin < 12 || isempty(flag_zh)
    flag_zh  = false;
end
if nargin < 13 || isempty(periodicity)
    periodicity  = zeros(1,d);
end
if nargin < 14 || isempty(tt_tol)
    tt_tol  = 1e-6;
end

% finding appropriate size for the embedding (approximately
% ensuring positive definiteness of embedded covariance matrix)
switch model_Y
    case 'matern'
        min_emb    = 3+micro_Y;
    case 'gaussian'
        min_emb    = 3+micro_Y;
    case 'exponential'
        min_emb    = 5+micro_Y;
    case 'spherical'
        min_emb    = 1+micro_Y;
    otherwise
        error('no valid geostatistical model chosen!')
end

% defining embedded domain size
n_ele_Y        = ceil(max(domain_len,min_emb*lambda_Y)./el_len + min_emb*lambda_Y./el_len);

% expanding embedded domain size for better factor decomposition
for i=1:d
    while 1==1
        factors = factor(n_ele_Y(i));
        nice    = prod(factors(factors<=5));
        bad     = n_ele_Y(i)/nice;
        if n_ele_Y(i)==nice, break, end
        better  = bad + 1;
        n_ele_Y(i) = nice*better;
    end
end

% enforcing periodicity, if desired by flag
n_ele_Y(periodicity==1) = n_el(periodicity==1);

% final definition of periodic/embedded domain
nele_Y         = prod(n_ele_Y);
domain_lene_Y  = n_ele_Y.*el_len;
%n_add_Y        = n_ele_Y - n_el;
%domain_add_Y   = domain_lene_Y-domain_len;

% defining embedded domain coordinates
h_effe = cell(1,d);
for i=1:d
    xele     = (0:n_ele_Y(i)-1)'*el_len(i);
    xele         = min(xele,domain_lene_Y(i)-xele);
    xele         = sqrt((domain_lene_Y(i)/2).^2+lambda_Y(i).^2)-lambda_Y(i)  -  (sqrt((domain_lene_Y(i)/2-xele).^2+lambda_Y(i).^2)-lambda_Y(i));
    xele = tt_tensor((xele/lambda_Y(i)).^2);
    h_effe{i} = xele;
end
h_effe = tt_meshgrid_vert(h_effe);
% Compute sum of squares by ALS -- will be rank-2 anyway
h_effe = amen_sum(h_effe, ones(d,1), 1e-12);
% Square root by TT cross
h_effe = amen_cross_s({h_effe}, @(h)sqrt(abs(h) + micro_Y.^2) - micro_Y, tt_tol);

    
% defining embedded covariance matrix
switch model_Y
    case 'matern'
%        Qse_Y        = variance_Y * amen_cross_s({h_effe},  @(h)matern_covariance(nu, h', [], ell, sigma ), tt_tol);
        Qse_Y        = variance_Y * amen_cross_s({h_effe},  @(h)matern_covariance(0.5, h', [], 0.5, 0.5 )', tt_tol);
    case 'gaussian'
        Qse_Y        = variance_Y * amen_cross_s({h_effe}, @(h)exp(-h.^2), tt_tol);
    case 'exponential'
        Qse_Y        = variance_Y * amen_cross_s({h_effe}, @(h)exp(-h), tt_tol);
    case 'spherical'
        Qse_Y        = variance_Y * amen_cross_s({h_effe}, @(h)(1 - 1.5*h + 0.5*h.^3).*double(h<=1), tt_tol);
end

% computing FFTn and eigenvalues of embedded covariance matrix
FFTQse_Y = core(Qse_Y);
for i=1:d
    % 1D FFT over the TT block
    FFTQse_Y{i} = fft(FFTQse_Y{i}, [], 1);
end
FFTQse_Y = tt_tensor(FFTQse_Y);
sqrtLambda = amen_cross_s({FFTQse_Y}, @(f)sqrt(abs(f)/nele_Y), tt_tol);

% loop to ensure non-NaN and non-Inf fields
while 1==1
    
    % computing embedded random field
    if flag_kit == false    % dietrich and newsam method 
                            % using rank-1 random field                            
        
        % Product of Gaussian vectors
        epsilon = cell(d,1);
        for i=1:d
            epsilon{i} = complex(randn(n_ele_Y(i),1), randn(n_ele_Y(i),1));
        end                
        epsilon = tt_tensor(epsilon);
        
        % Multiply with sqrt(lambda) and take inverse FFT
        Ye = amen_cross_s({sqrtLambda, epsilon}, @(x)prod(x,2), tt_tol, 'y0', sqrtLambda);
        Ye = core(Ye);
        for i=1:d            
            Ye{i} = ifft(Ye{i}, [], 1);
        end
        Ye = tt_tensor(Ye);
        Ye = real(Ye*nele_Y);
    else                    % kitanidis method
        error('flag_kit>0 is not implemented');
        epsilon        = exp(1i*angle(fftn(randn(n_ele_Y))));
        epsilon(1)     = 0;   % total MUST because abs(epsilon(1)) is always unity!
        Ye             = real(ifftn(epsilon.*sqrt(lambda)))*nele_Y;
    end

    if flag_zh > 0          % producing connected fields (zinn and harvey)
        error('flag_zh>0 is not implemented');
        Ye             = Ye/sqrt(variance_Y);  % ensuring zero mean unit variance
        % do not touch the mean value!
        % Ye must be zero mean in the ensemble, but enforcing zero mean
        % in single non-ergodic realizations messes up the mean after transformation!
        if flag_zh == 1       % low -K inclusions
            Ye             = -erfinv(2*erf(abs(Ye)*sqrt(0.5))-1)*sqrt(2);
        elseif flag_zh == 2   % high-K inclusions
            Ye             = erfinv(2*erf(abs(Ye)*sqrt(0.5))-1)*sqrt(2);
        else
            error('MC:GENERATE_Y:ZinnHarvey:incorrect_flag','Value of flag_zh must be 0, 1 or 2.')
        end
        Ye             = Ye*sqrt(variance_Y);  % re-installing desired variance
    end

    % extracting random field from embedded field
%     Y              = extraction(Ye,n_ele_Y,1,n_el);
    Y = core(Ye);
    for i=1:d
        Y{i} = Y{i}(1:n_el(i), :, :);
    end
    Y = tt_tensor(Y);

    if flag_kit == 2
        error('flag_kit=2 is not implemented');
        % ensuring a correct normal distribution of mean values
        % !must be used for Kitanidis method and extracted fields!
        % otherwise, mean value has a bimodal distribution.
        % !! leads to spatially varying variance of Y !!
        % varmeanY     = E[u'*(Y-pertY)*(Y-pertY)'*u] = u'*CovYY*u;
        u              = ones(nel,1)/nel;
        ue             = embedding(u,n_el,1,n_ele_Y);
        Qse_ue         = circ_QHT(FFTQse_Y,ue,n_ele_Y);
        Qsu            = extraction(Qse_ue,n_ele_Y,1,n_el);
        uQsu           = u'*Qsu;
        Y              = Y - mean(Y(:)) + randn(1,1)*sqrt(uQsu);
    end

    % adding (uncertain) mean value
    if det(Qbb_Y) == 0 % known mean
        Y     = Y + beta_Y;
    else % uncertain mean
        beta  = beta_Y + chol(Qbb_Y)'*randn(size(Qbb_Y,1),1);
        Y     = Y + tt_ones(Y.n)*beta;
    end

%     % exit only when all non-NaN and all non-Inf
%     if ~any(isnan(Y)) && all(Y~=Inf) && all(Y~=-Inf), break; end
%     fprintf( 'something is wrong here' );
    break
end

% ----------------------------------------------------------
%   S U B F U N C T I O N   1
% ----------------------------------------------------------

function Me = embedding(M,n_,nmeas,n_e)
% embeds a (nmeas x prod(n_)) matrix into a (nmeas x prod(n_e)) matrix
% version 15 march 2006 / WN

% M       : matrix size nmeas x prod(n_) (to be embedded)
% n_      : size of original domain in y,x,z direction
% nmeas   : number of measurements = numer of lines in M
% n_e     : size of periodic embedding domain in y,x,z direction

Me = zeros([nmeas, n_e]);
if length(n_)==2
    Me(:,1:n_(1),1:n_(2))  = reshape(M,nmeas,n_(1),n_(2));
elseif length(n_)==3
    Me(:,1:n_(1),1:n_(2),1:n_(3))  = reshape(M,nmeas,n_(1),n_(2),n_(3));
end
Me = reshape(Me,nmeas,prod(n_e));

% ----------------------------------------------------------
%   S U B F U N C T I O N   2
% ----------------------------------------------------------

function M = extraction(Me,n_e,nmeas,n_)
% extracts a (nmeas x ny*nx) matrix from a (nmeas x nye*nxe) embedding matrix
% version 15 march 2006 / WN

% M       : matrix size prod(n_ ) x meas (to be extracted)
% Me      : matrix size prod(n_e) x nmeas (from which M is to be extracted)
% n_      : size of original domain in y,x,z direction
% nmeas   : number of measurements = numer of rows in M
% n_e     : size of periodic embedding domain in y,x,z direction

Me = reshape(Me,[n_e, nmeas]);
if length(n_)==2
    M  = reshape(Me(1:n_(1),1:n_(2),:),prod(n_),nmeas);
elseif length(n_)==3
    M  = reshape(Me(1:n_(1),1:n_(2),1:n_(3),:),prod(n_),nmeas);
end

% ----------------------------------------------------------
%   S U B F U N C T I O N   3
% ----------------------------------------------------------

function cQHT = circ_QHT(FFT_Q,H,n_e)
% CIRC_QHT computes Q*H' with circulant matrix from first row
% version 15 march 2006 / WN

nk          = size(H,1);
cQHT        = zeros(nk,prod(n_e));

for i=1:nk
    V         = real(ifftn(fftn(reshape(H(i,:),n_e)).*FFT_Q));
    cQHT(i,:) = reshape(V,1,prod(n_e));
end




