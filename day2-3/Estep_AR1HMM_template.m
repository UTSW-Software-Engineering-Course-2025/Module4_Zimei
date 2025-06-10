function [c, gammaMat, xiArr] = ...
            Estep_AR1HMM_template(x, x1, initPr, tranPr, phi0, phi1, sigmasq, T, M)
% Estep_AR1HMM
%
% J Noh, 2025/02

%% Define objects
alphaMat = zeros(T, M);
betaMat = zeros(T, M);
gammaMat = zeros(T, M);
xiArr = zeros(T-1, M, M);
bMat = zeros(T, M); 

c = zeros(T, 1);              % normalizing scale factor for alpha_t(i)
d = zeros(T, 1);              % normalizing scale factor for beta_t(i)

%% pdf function values, b_i(x_t | x_(t-1))
for i = 1:M
    mu = phi0(i) + phi1(i) * x1(:);
    sigma = ( sigmasq(i) )^0.5;
    bMat(:, i) = pdf('Normal', x(:), mu(:), sigma);
end

%% alpha_t(i) forward equation
% Calculation of the forward probabilities (alpha) with scaling to prevent underflow.
alphaMat(1, :) = initPr(:)' .* bMat(1, :);
c(1) = sum(alphaMat(1, :));
alphaMat(1, :) = alphaMat(1, :) / c(1);

for t = 2:T
    alphaMat(t, :) = (alphaMat(t-1, :) * tranPr) .* bMat(t, :);
    c(t) = sum(alphaMat(t, :));
    if c(t) > 0
        alphaMat(t, :) = alphaMat(t, :) / c(t);
    end
end


%% beta_t(j) backward equation
betaMat(T, :) = 1;
d(T) = 1;
for t = T-1:-1:1
    betaMat(t, :) = (betaMat(t+1, :) .* bMat(t+1, :)) * tranPr';
    d(t) = c(t+1);
    if d(t) > 0
        betaMat(t, :) = betaMat(t, :) / d(t);
    end
end


%% define gamma_t(i) 
gammaMat = alphaMat .* betaMat;
% Renormalize for numerical stability
gammaMat = gammaMat ./ sum(gammaMat, 2);

%% define xi_t(i,j)  
for t = 1:T-1
    if c(t+1) > 0
        % Calculate the numerator of the xi formula
        numerator = (alphaMat(t, :)' * (bMat(t+1, :) .* betaMat(t+1, :))) .* tranPr;
        % Normalize to get the probability
        xiArr(t, :, :) = numerator / sum(numerator(:));
    end
end


end
