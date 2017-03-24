function [seq] = sample_GP(x, sigma, xc, nb)
% Use fft to sample Gaussian processes

N = size(x, 1);
seq = zeros(nb, N);
R = sigma^2 * exp(-x.^2 / xc^2)';
filter = fft(fftshift(R));

for i = 1:nb
    W = randn(1, N);
    F = ifft(sqrt(filter) .* fft(W));
    seq(i, :) = real(F);
end 

end

