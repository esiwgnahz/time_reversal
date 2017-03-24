function [U] = split_step_fourier_method(start, step, ending, u0, h, k, grid_f, GP_seq)

    u = u0; % Initial condition
    grid_f = fftshift(grid_f');
    c = fft(u); % Go to Fourier Space

    for m = start:step:ending % Start time loop
        c = exp(-h/2/k*1i*(grid_f).^2).*c; % Advance in Fourier Space
        u = ifft(c); % Go to spatial Space
        if nargin > 7 % If random medium
            mu = GP_seq(floor(abs(h)*m)+1,:)'; % Gaussian process sample
            u = exp(h/2*1i*k.*mu).*u; % Solve non-constant part of LSE
        end
        c = fft(u); % Go to Fourier Space
    end
    U = ifft(c); % Go finaly to spatial Space
end




