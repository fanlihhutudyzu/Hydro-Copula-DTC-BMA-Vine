function [sim_data, model_info] = gaussian_t_copula_model(U, m)
% gaussian_t_copula_model  — fit Gaussian and t copula to input U and simulate
% INPUTS:
%   U : n x d matrix of samples on (0,1) scale (pseudo-observations)
%   m : number of simulated samples to generate
% OUTPUTS:
%   sim_data : m x d simulated samples on (0,1) scale
%   model_info : struct with fields {R_gauss, R_t, nu_t, BIC_Gauss, BIC_t, chosen_type}

[n,d] = size(U);

% Fit Gaussian copula
R_gauss = copulafit('Gaussian', U);  % returns correlation matrix

% Compute log-likelihood and BIC for Gaussian
ll_gauss = sum(log(copulapdf('Gaussian', U, R_gauss)));
k_gauss = d*(d-1)/2;  % number of free parameters in correlation matrix
BIC_Gauss = -2*ll_gauss + k_gauss*log(n);

% Fit t copula
% copulafit for 't' returns [Rhat,nuhat]
try
    [R_t, nu_t] = copulafit('t', U);
catch ME
    warning(ME.identifier,'t copula fit failed: %s. Trying default nu=4 and estimate R by inversion.', ME.message);
    nu_t = 4;
    R_t = copulafit('Gaussian', U); % fallback
end

% Compute log-likelihood and BIC for t
ll_t = sum(log(copulapdf('t', U, R_t, nu_t)));
k_t = d*(d-1)/2 + 1; % +1 for nu
BIC_t = -2*ll_t + k_t*log(n);

% Choose best
if BIC_t < BIC_Gauss
    chosen = 't';
    params = struct('R', R_t, 'nu', nu_t);
else
    chosen = 'Gaussian';
    params = struct('R', R_gauss);
end

% Simulate m samples from chosen copula
if strcmp(chosen,'t')
    Usim = copularnd('t', params.R, params.nu, m); 
else
    Usim = copularnd('Gaussian', params.R, m);
end

% Output
sim_data = Usim;
model_info = struct('R_gauss',R_gauss,'R_t',R_t,'nu_t',nu_t,'BIC_Gauss',BIC_Gauss,'BIC_t',BIC_t,'chosen_type',chosen,'chosen_params',params);
end