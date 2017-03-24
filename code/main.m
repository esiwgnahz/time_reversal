%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Propagation of time-harmonic waves in a random medium %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% PART I - Homogeneous medium %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

xmax = 60;
k = 1; 
w = 1; % Frequency
L = 10; % Position of mirror
r0 = 2; % Gaussian beam radius
N = 2^10 + 1;
x = linspace(-xmax/2, xmax/2, N);

%% Part 1 : Propagation of time-harmonic waves in a homogeneous medium

% theoretical part 
Rt = r0  * (1 + 4 * L^2/k^2/r0^4)^0.5;
psi_t = r0/Rt * exp(-2 * x.^2./Rt^2); 

% simulation 
psi_0 = exp(-x.^2/r0^2);
dx = xmax /(N - 1);
psi_0_fft = fftshift(fft(psi_0));

fmax = 1/2/dx;
f = 2 * pi .* linspace(-fmax, fmax, N ); 

sol_L_fft = psi_0_fft .* exp(-1i * f.^2/2/k * L); 
sol = ifft(sol_L_fft);
sol_norm = abs(sol).^2 ;

figure(1); plot(x, psi_t, '-r', x, sol_norm, '-b', x, psi_0, '-k');
legend('Theoretical solution on z = L', 'Simulation on z = L', 'Wave on z = 0')
xlabel('x'); ylabel('Square of the amplitude');grid('on')
title('Wave on z = L');

%% Part 1 : Time reversal for time-harmonic waves in a homogeneous medium
rt = r0* (1 + 2i * L/k/r0^2)^0.5;
psi_L = r0/rt * exp(-x.^2/rt^2);
psi_L_conj = conj(psi_L);

% chi_M compactly supported
rM = 2:4:22;
sol_0 = zeros(length(rM), length(x));

for i = 1 : length(rM)
    chi_M = (1 - (x./2/rM(i)).^2).^2;
    chi_M(x <-2*rM(i)) = 0 ;
    chi_M(x > 2*rM(i)) = 0 ;

    term1_fft = fftshift(fft(psi_L_conj .* chi_M));
    sol_0_fft = term1_fft.* exp(1i * f.^2/2/k * L);
    sol_0(i, :) = abs(ifft(sol_0_fft));
end
figure(2); plot(x, sol_0, 'LineWidth', 2);
legend('rM = 2',  'rM = 6',  'rM = 10',  'rM = 14', 'rM = 18', 'rM = 22')
title('Wave on z = 0, time reversal')
xlabel('x'); ylim([min(min(sol_0)), max(max(sol_0))]); ylabel('Amplitude'); grid('off')

% chi_M gaussian
rM = 2;
chi_M = exp(- (x.^2)/rM^2);
term1_fft = fftshift(fft(psi_L_conj .* chi_M));
sol_0_fft = term1_fft.* exp(1i * f.^2/2/k * L);
sol_0 = abs(ifft(sol_0_fft));
atr = (1 -4i * L/k/r0^2 -  4 * L^2/k^2/r0^2/rM^2 - 2i*L/k/rM^2)^0.5;
rtr_square = (1/rM^2 + 1/(r0^2 - 2i * L/k))^(-1) - 2i * L/k;
psi_tr_0 = abs(1/atr * exp(- x.^2/rtr_square));

figure(3); plot(x, sol_0, '-r', x, psi_tr_0, '-b');
legend('Simulation', 'Theoretical solution')
xlabel('x'); ylabel('Amplitude'); grid('on')
title('Wave on z = 0, Gaussian time-reversal mirror')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Propagation of time-harmonic waves in a random medium %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% PART II - Heterogeneous medium %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Constants
h = 1; % Longitudinal (along z) step size
zc = 1;
xc = 4;
sigma = 1;

x = linspace(-xmax/2, xmax/2, N)';
dx = xmax / (N - 1);
fmax = 1 / (2 * dx);
f = 2 * pi .* linspace(-fmax, fmax, N ); 

u0 = exp(-x.^2/r0^2); % Initial wave profile
rM = 2;
mirror = exp(-x.^2 / rM^2); % Gaussian mirror profile

%% Part 2 : Runs propagations and backpropagations

nb_MC = 300; % Number of profiles to be averaged
nb_GP = round(L / zc); % Number of Gaussian processes

U_L = zeros(nb_MC, N); % Transmitted wave profile
U_0_rand = zeros(nb_MC, N); % Refocused wave profile random medium
U_0_homo = zeros(nb_MC, N); % Refocused wave profile homogeneous medium

for i = 1:nb_MC  
    % Forward propagation in random medium
    GP_seq = sample_GP(x, sigma, xc, nb_GP);
    U_L(i,:) = split_step_fourier_method(0, 1, round(L/h) - 1, u0, h, k, f, GP_seq);
    
    % Backpropagation in same random medium
    u = conj(U_L(i,:)') .* mirror;
    U_0_rand(i,:) = split_step_fourier_method(round(L/h) - 1, -1, 0, u, -h, k, f, GP_seq);
    
    % Backpropagation in homogeneous medium
    U_0_homo(i,:) = split_step_fourier_method(round(L/h) - 1, -1, 0, u, h, k, f);
end

% Average results on all runs
U_L = mean(U_L, 1)';
U_0_rand = mean(U_0_rand, 1)';
U_0_homo = mean(U_0_homo, 1)';

%% Part 2 : Mean transmitted wave profile
rt = r0*sqrt(1 + 2*1i*L/k/r0^2);
gamma0 = sigma^2*zc;
mean_wave_L = r0/rt * exp(-x.^2/rt^2) * exp(-gamma0*w^2*L/8);

figure(4); plot(x, abs(mean_wave_L), x, abs(U_L))
xlabel('x'); ylabel('|\bf{E}[\phi_t(x)]|');
legend('theoretical', 'empirical')
title('Mean transmitted wave profile in random medium')

%% Part 2 : mean refocused wave profile in z=0 (random medium)
atr = sqrt(1 + 4*L^2/(k*r0*rM)^2 + 2*1i*L/k/rM^2);
rtr_square = ((1/rM^2+1/(r0^2-2*1i*L/k))^-1 + 2*1i*L/k);
gamma2 = 2*sigma^2*zc/xc^2;
ra_square = 48/L/gamma2/w^2  * 1;
mean_wave_0_rand = 1/atr * exp(-x.^2 /rtr_square) .* exp(-x.^2 /ra_square);

figure(5); plot(x, 0.28*abs(mean_wave_0_rand), x, 0.28*abs(U_0_rand));
xlabel('x'); ylabel('|\bf{E}[\phi_t^{tr}(x)]|');
legend('theoretical', 'empirical')
title('Mean refocused wave profile in random medium')

%% Part 2 : mean refocused wave profile in z=0 (homogeneous medium)
mean_wave_0_homo = 1/atr * exp(-x.^2/rtr_square) .* exp(-gamma0*w^2*L/8);
figure(6); plot(x, abs(mean_wave_0_homo), x, abs(U_0_homo));
xlabel('x'); ylabel('|\bf{E}[\phi_t^{tr}(x)]|');
legend('theoretical', 'empirical')
title('Mean refocused wave profile in homegeneous medium')

%% Part 2 : Time reversal for time-dependent waves in a random medium.
w0 = 1;
B = 0.75;
nb_discr = 20;
omega_discr = linspace(w0-B,w0+B,nb_discr);
nb_GP = round(L / zc); % Number of Gaussian processes

U_0_w_samples = zeros(5, N);

for s = 1:5

U_L_w = zeros(nb_discr, N); % Transmitted wave profile
U_0_w = zeros(nb_discr, N); % Refocused wave profile random medium

GP_seq = sample_GP(x, sigma, xc, nb_GP);
for wi = 1:nb_discr
    %omega = omega_discr(w);
    k = omega_discr(wi);

    % Go forward in random medium
    U_L_w(wi,:) = split_step_fourier_method(0, 1, round(L/h) - 1, u0, h, k, f, GP_seq);

    % Go reverse in same random medium
    u = conj(U_L_w(wi,:)') .* mirror;
    U_0_w(wi,:) = split_step_fourier_method(round(L/h) - 1, -1, 0, u, -h, k, f, GP_seq);
end
U_0_w = mean(U_0_w, 1);
U_0_w_samples(s,:) = abs(U_0_w);
end

atr = sqrt(1 + 4*L^2/(k*r0*rM)^2 + 2*1i*L/k/rM^2);
rtr_square =(1/rM^2+1/(r0^2-2*1i*L/k))^-1 + 2*1i*L/k;
gamma2 = 2*sigma^2*zc/xc^2;
ra_square = 48/(L*gamma2*w^2);
mean_wave_0_rand = 1/atr * exp(-x.^2/rtr_square) .* exp(-x.^2/ra_square);

figure(7); 
plot(x, abs(mean_wave_0_rand), x, U_0_w_samples);
xlabel('x'); ylabel('|\bf{E}[\phi_t^{tr}(x)]|');
legend('theoretical')
title('Refocused time-dependent waves in random medium')