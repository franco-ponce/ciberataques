%% ===============================================================
%  TIME SERIES → IMAGE ENCODINGS (New methods)
%  Requires:
%   - X : time series vector (row or column)
%   - fs: sampling frequency (Hz)
%  Output:
%   - img_spec   : Spectrogram (STFT)
%   - img_scalo  : Scalogram (CWT)
%   - img_gray   : Gray-scale encoding
%   - img_ssm    : Self-similarity matrix (distance matrix)
% ===============================================================

%% --- Input data ---
% Assumes X is already in the workspace.
% If you don't have fs, you can set fs = 1; (arbitrary time unit)
fs = 1;   % <-- CHANGE this according to your signal

X = X(:);             % Ensure column vector
N = numel(X);

%% --- Auxiliary normalization (0-1) ---
X_min = min(X);
X_max = max(X);
X_norm = (X - X_min) / (X_max - X_min + eps);

%% ===============================================================
% 1) SPECTROGRAM (STFT)
%    Time-frequency image using a sliding window
% ===============================================================

% STFT parameters (adjust them for the desired resolution)
win_len   = min(256, max(64, floor(N/8)));  % window size
win       = hamming(win_len);
noverlap  = floor(0.5 * win_len);
nfft      = 2^nextpow2(win_len);

[S, F, T] = spectrogram(X, win, noverlap, nfft, fs);
img_spec  = abs(S);   % magnitude (you can use abs(S).^2 for power)

%% ===============================================================
% 2) SCALOGRAM (CWT)
%    Time-frequency image based on continuous wavelets
%    Requires Wavelet Toolbox
% ===============================================================

% cfs: complex wavelet coefficients (freq x time)
% freq: associated frequencies
[cfs, freq] = cwt(X, fs);
img_scalo   = abs(cfs);   % coefficient magnitude

%% ===============================================================
% 3) GRAY-SCALE ENCODING
%    Rearrange the normalized series into a 2D image
% ===============================================================

% We look for a square image of size m x m
m        = floor(sqrt(N));
X_crop   = X_norm(1 : m^2);       % crop to get m^2 samples
img_gray = reshape(X_crop, m, m); % 2D matrix in [0,1]

%% ===============================================================
% 4) SELF-SIMILARITY MATRIX (SELF-SIMILARITY / DISTANCE MATRIX)
%    Image where each pixel is the distance |x_i - x_j|
%    To avoid huge matrices, we apply a maximum sampling size.
% ===============================================================

maxN = 256;  % maximum matrix size (adjust if you want)
if N > maxN
    idx = round(linspace(1, N, maxN));
    X_ssm = X_norm(idx);
else
    X_ssm = X_norm;
end
M = numel(X_ssm);

% Distance matrix |x_i - x_j|
img_ssm = zeros(M, M);
for i = 1:M
    img_ssm(i, :) = abs(X_ssm(i) - X_ssm(:)).';
end

%% ===============================================================
% VISUALIZATION (optional)
% ===============================================================

figure('Name','Time Series Imaging','Color','w');

% 1) Spectrogram
subplot(2,2,1);
imagesc(T, F, img_spec);
axis xy; xlabel('Tiempo'); ylabel('Frecuencia (Hz)');
title('Espectrograma (STFT)');
colorbar;

% 2) Scalogram
subplot(2,2,2);
imagesc((1:size(img_scalo,2))/fs, freq, img_scalo);
axis xy; xlabel('Tiempo'); ylabel('Frecuencia (Hz)');
title('Escalograma (CWT)');
colorbar;

% 3) Gray-scale encoding
subplot(2,2,3);
imagesc(img_gray);
axis image xy; xlabel('col'); ylabel('fila');
title('Codificación en escala de grises');
colorbar;

% 4) Self-Similarity Matrix
subplot(2,2,4);
imagesc(img_ssm);
axis image xy;
xlabel('t_j'); ylabel('t_i');
title('Matriz de auto-similitud |x_i - x_j|');
colorbar;
