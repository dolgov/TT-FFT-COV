% Applies linear fun (MV, fft) to each variable in TT_tensor x
% Input:
%   x: input tt_tensor
%   fun: function that must be applied along rows of each TT block reshaped
%        into an n x (r1*r2) matrix.
% Output:
%   y: resulting tt_tensor
function [y] = tt_modefun(x, fun)

y = core(x);
for i=1:numel(y)
    [n,r1,r2] = size(y{i});
    y{i} = reshape(y{i}, n, r1*r2);
    y{i} = fun(y{i});
    y{i} = reshape(y{i}, [], r1, r2);
end
y = tt_tensor(y);

end
