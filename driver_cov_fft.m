%% Simple low-rank kriging of 2D moisture data using circulant covariance matrices


% Load sampled data
zs = load('moist_64000_31.txt');
xs = zs(:,1);
ys = zs(:,2);
zs = zs(:,3);
figure(1); scatter3(ys,xs,zs,1,zs); view(2); colorbar; title('samples');

model_Y     = 'exponential';  % type of covariance function
ncoarse = 129;                % univariate size of coarse grid
nfine = 513;                  % univariate size of fine grid
tol_cov = 1e-3;               % TT tolerance for covariance column
tol_y = 2e-1;                 % TT tolerance for fields
nugget = 0;                   % shift in C_{yy}


%% Regress data onto coarse rectangular grid
xcoarse = linspace(floor(min(xs))+1,ceil(max(xs))-1,ncoarse)';
ycoarse = linspace(floor(min(ys))+1,ceil(max(ys))-2,ncoarse)';

tic;
y = scatteredInterpolant(xs,ys,zs,'linear');
y = y(repmat(xcoarse,1,ncoarse), repmat(ycoarse',ncoarse,1));
y = tt_tensor(y, tol_y);
fprintf('Regression took %g sec., rank(y) = %d\n\n', toc, max(y.r));
figure(2); mesh(ycoarse,xcoarse,full(y,y.n')); view(2); colorbar; title('coarse (linearly interpolated)');


%% Exponential cov on fine grid (first column)
xfine = linspace(floor(min(xs))+1,ceil(max(xs))-1,nfine)';
yfine = linspace(floor(min(ys))+1,ceil(max(ys))-2,nfine)';

X = tt_meshgrid_vert({tt_tensor(xfine-xfine(1)), tt_tensor(yfine-yfine(1))});

% Choose the covariance function
switch model_Y
    case 'matern'
        Cfun = @(x)matern_covariance(0.5, sqrt(sum(x.^2, 2))', [], 0.5, 0.5 )';
    case 'gaussian'
        Cfun = @(x)exp(-sum(x.^2, 2)/2);
    case 'exponential'
        Cfun = @(x)exp(-sqrt(sum(x.^2, 2))/0.5);
    case 'spherical'
        Cfun = @(x)(1 - 1.5*x + 0.5*x.^3).*double(x<=1);
end

tic;
C_fine = amen_cross_s(X, Cfun, tol_cov);
fprintf('Cov column took %g sec., rank(C_{ss}) = %d\n\n', toc, max(C_fine.r));

% Fourier image
C_fine_f = tt_modefun(C_fine, @fft);

% Coarse grid subsampling matrix
H = speye(nfine);
step = (nfine-1)/(ncoarse-1);
H = H(1:step:nfine, :);

% Coarse cov
C_coarse = tt_modefun(C_fine, @(c)c(1:step:nfine, :));

% Its Fourier image
C_coarse_f = tt_modefun(C_coarse, @fft);


%%  Kriging
% C_{yy}^{-1}
tic;
xi = tt_modefun(y, @fft);
xi = amen_cross_s({xi,C_coarse_f}, @(x)x(:,1)./(x(:,2)+nugget), tol_y);
xi = tt_modefun(xi, @ifft);
% Embed into full space
s = tt_modefun(xi, @(s)H'*s);
% Multiply C_fine via FFT
s = tt_modefun(s, @fft);
s = amen_cross_s({s,C_fine_f}, @(x)prod(x,2), tol_y);
s = tt_modefun(s, @ifft);
fprintf('Kriging took %g sec., rank(xi) = %d, rank(s) = %d\n\n', toc, max(xi.r), max(s.r));

figure(3); mesh(yfine,xfine,real(full(s,s.n'))); view(2); colorbar; title('fine(kriged)');

