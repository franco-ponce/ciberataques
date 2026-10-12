function [T, dT, ddT] = timeseries_to_images_complete(X, fs, timestamps)
% TIMESERIES_TO_IMAGES_COMPLETE Converts time series into multiple image representations
%
% Computes 8 conversion methods on the original signal and its derivatives (1st and 2nd)
% Generates 24 images in total: 8 methods × 3 signals (X, dX, ddX)
%
% IMPLEMENTED METHODS:
%   1. GASF  - Gramian Angular Summation Field
%   2. GADF  - Gramian Angular Difference Field
%   3. MTF   - Markov Transition Field
%   4. RP    - Recurrence Plot
%   5. SPEC  - Spectrogram (STFT)
%   6. SCALO - Scalogram (CWT)
%   7. GRAY  - Gray-scale encoding
%   8. SSM   - Self-Similarity Matrix
%
% INPUTS:
%   X         - Time series vector (column or row)
%   fs        - Sampling frequency in Hz (optional, default = 1)
%   timestamps- Timestamp vector (optional)
%
% OUTPUTS:
%   T(:,:,1:8)   - 8 images of the original signal X
%   dT(:,:,1:8)  - 8 images of the first derivative dX
%   ddT(:,:,1:8) - 8 images of the second derivative ddX
%
% INDICES:
%   (:,:,1) = GASF
%   (:,:,2) = GADF
%   (:,:,3) = MTF
%   (:,:,4) = RP
%   (:,:,5) = Spectrogram
%   (:,:,6) = Scalogram
%   (:,:,7) = Gray-scale encoding
%   (:,:,8) = Self-Similarity Matrix
%
% EXAMPLE:
%   X = sin(linspace(0, 4*pi, 200))' + 0.1*randn(200,1);
%   fs = 100; % Hz
%   [T, dT, ddT] = timeseries_to_images_complete(X, fs);
%
%   % Plot GASF of the 3 signals
%   figure;
%   subplot(1,3,1); imagesc(T(:,:,1)); title('GASF - Original');
%   subplot(1,3,2); imagesc(dT(:,:,1)); title('GASF - 1st Derivative');
%   subplot(1,3,3); imagesc(ddT(:,:,1)); title('GASF - 2nd Derivative');
%
% Author: Complete unified script
% Date: November 2025
% Version: 3.0

    %% ====================================================================
    %  INPUT VALIDATION AND PREPARATION
    %% ====================================================================

    % Argument validation
    if nargin < 2 || isempty(fs)
        fs = 1;  % Default sampling frequency
    end

    if nargin < 3
        timestamps = [];
    end

    % Make sure X is a column vector
    if isrow(X)
        X = X(:);
    end

    % Basic parameters
    N = length(X);

    if N < 3
        error('La serie de tiempo debe tener al menos 3 puntos.');
    end

    %% ====================================================================
    %  DERIVATIVE COMPUTATION WITH SIZE ADJUSTMENT
    %% ====================================================================

    % First derivative: dX(1) = X(1), rest = diff(X)
    dX = zeros(N, 1);
    dX(1) = X(1);
    dX(2:N) = diff(X);

    % Second derivative: ddX(1) = X(1), ddX(2) = dX(2), rest = diff(dX)
    ddX = zeros(N, 1);
    ddX(1) = X(1);
    ddX(2) = dX(2);
    ddX(3:N) = diff(dX(2:N));

    %% ====================================================================
    %  OUTPUT MATRIX INITIALIZATION
    %% ====================================================================

    % Determine max size to make all images uniform
    % We use N×N for most, but some methods may vary
    max_size = N;

    T = zeros(max_size, max_size, 4);    % Original signal
    dT = zeros(max_size, max_size, 4);   % First derivative
    ddT = zeros(max_size, max_size, 4);  % Second derivative

    %% ====================================================================
    %  PROCESSING: APPLY ALL METHODS TO THE 3 SIGNALS
    %% ====================================================================

    %fprintf('Procesando conversiones de series de tiempo a imágenes...\n');
    %fprintf('Longitud de serie: N = %d\n', N);
    %fprintf('Frecuencia de muestreo: fs = %.2f Hz\n\n', fs);

    % Process original signal
    %fprintf('[1/3] Procesando señal original...\n');
    T = process_single_timeseries(X, fs, N);

    % Process first derivative
    %fprintf('[2/3] Procesando primera derivada...\n');
    dT = process_single_timeseries(dX, fs, N);

    % Process second derivative
    %fprintf('[3/3] Procesando segunda derivada...\n');
    ddT = process_single_timeseries(ddX, fs, N);

    fprintf('\n✓ Procesamiento completado exitosamente.\n');
    fprintf('  Total de imágenes generadas: %d (4 métodos × 3 señales)\n', 8*3);

    % Timestamp processing (optional, for future extensions)
    if ~isempty(timestamps)
        try
            DateNumber = datenum(timestamps);
            % Additional timestamp-based logic can be added here
        catch
            warning('No se pudo procesar timestamp. Continuando sin él.');
        end
    end
end

%% ========================================================================
%% HELPER FUNCTION: PROCESS A SINGLE TIME SERIES
%% ========================================================================

function img_stack = process_single_timeseries(X, fs, target_size)
    % Processes a time series with all methods
    % Returns a stack of 8 images

    N = length(X);
    img_stack = zeros(target_size, target_size, 4);

    % Normalize signal to [0, 1]
    X_norm = normalize_to_unit(X);

    %% METHODS 1-4: GRAMIAN AND OTHER ALGEBRAIC METHODS

    % 1. GASF - Gramian Angular Summation Field
    % 2. GADF - Gramian Angular Difference Field
    [gasf, gadf] = compute_gramian_fields(X_norm);
    img_stack(:,:,1) = resize_to_target(gasf, target_size);
    img_stack(:,:,2) = resize_to_target(gadf, target_size);

    % 3. MTF - Markov Transition Field
    mtf = compute_markov_transition_field(X_norm, N);
    img_stack(:,:,3) = resize_to_target(mtf, target_size);

    % 4. RP - Recurrence Plot
    rp = compute_recurrence_plot(X_norm, N, 0);
    img_stack(:,:,4) = resize_to_target(rp, target_size);

    % %% METHODS 5-8: FREQUENCY AND TRANSFORM METHODS
    %
    % % 5. Spectrogram (STFT)
    % spec = compute_spectrogram(X, fs);
    % img_stack(:,:,5) = resize_to_target(spec, target_size);
    %
    % % 6. Scalogram (CWT)
    % scalo = compute_scalogram(X, fs);
    % img_stack(:,:,6) = resize_to_target(scalo, target_size);
    %
    % % 7. Gray-scale encoding
    % gray = compute_grayscale_encoding(X_norm);
    % img_stack(:,:,7) = resize_to_target(gray, target_size);
    %
    % % 8. Self-Similarity Matrix
    % ssm = compute_self_similarity_matrix(X_norm);
    % img_stack(:,:,8) = resize_to_target(ssm, target_size);

    % Clean up NaNs and Inf
    img_stack(isnan(img_stack) | isinf(img_stack)) = 0;
end

%% ========================================================================
%% HELPER FUNCTIONS: METHODS 1-4 (ORIGINAL)
%% ========================================================================

function X_norm = normalize_to_unit(X)
    % Normalizes vector X to the range [0, 1]
    X_min = min(X);
    X_max = max(X);

    if abs(X_max - X_min) < eps
        X_norm = zeros(size(X));
    else
        X_norm = (X - X_min) / (X_max - X_min);
    end
end

function [GASF, GADF] = compute_gramian_fields(X_norm)
    % Computes the Gramian Angular fields (GASF and GADF)

    N = length(X_norm);

    % Convert normalized values to angles
    theta = acos(X_norm);

    % Initialize matrices
    GASF = zeros(N, N);
    GADF = zeros(N, N);

    % Compute Gramian fields
    for j = 1:N
        for k = 1:N
            GASF(j, k) = cos(theta(j) + theta(k));
            GADF(j, k) = sin(theta(j) - theta(k));
        end
    end
end

function MTF = compute_markov_transition_field(V, L)
    % Computes the Markov Transition Field

    N = length(V);

    % Discretize vector V into L bins
    [~, ~, V_bins] = histcounts(V, L);

    % Handle the case where all values fall into a single bin
    if all(V_bins == 0)
        V_bins = ones(size(V_bins));
    end

    % Build the Markov transition matrix
    T = zeros(L, L);
    for i = 1:(N-1)
        if V_bins(i) > 0 && V_bins(i+1) > 0
            T(V_bins(i), V_bins(i+1)) = T(V_bins(i), V_bins(i+1)) + 1;
        end
    end

    % Normalize the transition matrix
    row_sums = sum(T, 2);
    for i = 1:L
        if row_sums(i) > 0
            T(i, :) = T(i, :) / row_sums(i);
        end
    end
    T(isnan(T)) = 0;

    % Generate the Markov Transition Field
    MTF = zeros(N, N);
    for i = 1:N
        for j = 1:N
            if V_bins(i) > 0 && V_bins(j) > 0
                MTF(i, j) = T(V_bins(i), V_bins(j));
            end
        end
    end
end

function RP_matrix = compute_recurrence_plot(timeSeries, m, tau)
    % Computes the Recurrence Plot

    N = length(timeSeries);

    % Build the phase-space reconstruction
    num_points = N - (m - 1) * tau;

    if num_points <= 0
        % If there are not enough points, use a simple distance matrix
        RP_matrix = pdist2(timeSeries(:), timeSeries(:));
        max_dist = max(RP_matrix(:));
        if max_dist > 0
            RP_matrix = RP_matrix / max_dist;
        end
        return;
    end

    phaseSpace = zeros(num_points, m);

    for i = 1:m
        start_idx = 1 + (i - 1) * tau;
        end_idx = N - (m - i) * tau;
        phaseSpace(:, i) = timeSeries(start_idx:end_idx);
    end

    % Compute distance matrix
    distanceMatrix = pdist2(phaseSpace, phaseSpace);

    % Normalize by the maximum distance
    max_dist = max(distanceMatrix(:));
    if max_dist > 0
        RP_matrix = distanceMatrix / max_dist;
    else
        RP_matrix = distanceMatrix;
    end
end

%% ========================================================================
%% HELPER FUNCTIONS: METHODS 5-8 (NEW)
%% ========================================================================

function img_spec = compute_spectrogram(X, fs)
    % Computes the Spectrogram using STFT

    N = length(X);

    % Adaptive STFT parameters
    win_len  = min(256, max(64, floor(N/8)));
    win      = hamming(win_len);
    noverlap = floor(0.5 * win_len);
    nfft     = 2^nextpow2(win_len);

    try
        [S, ~, ~] = spectrogram(X, win, noverlap, nfft, fs);
        img_spec = abs(S);  % Magnitude
    catch
        % If it fails, create a zero matrix
        warning('No se pudo calcular el espectrograma. Retornando matriz de ceros.');
        img_spec = zeros(nfft/2 + 1, ceil((N - noverlap) / (win_len - noverlap)));
    end

    % Normalize to [0, 1]
    max_val = max(img_spec(:));
    if max_val > 0
        img_spec = img_spec / max_val;
    end
end

function img_scalo = compute_scalogram(X, fs)
    % Computes the Scalogram using CWT (Continuous Wavelet Transform)

    try
        % Requires Wavelet Toolbox
        [cfs, ~] = cwt(X, fs);
        img_scalo = abs(cfs);  % Coefficient magnitude

        % Normalize to [0, 1]
        max_val = max(img_scalo(:));
        if max_val > 0
            img_scalo = img_scalo / max_val;
        end
    catch ME
        % If Wavelet Toolbox is not available or it fails
        warning('No se pudo calcular el escalograma: %s. Retornando matriz de ceros.', ME.message);
        N = length(X);
        img_scalo = zeros(min(128, N), N);
    end
end

function img_gray = compute_grayscale_encoding(X_norm)
    % Gray-scale encoding: rearrange the series into a 2D image

    N = length(X_norm);

    % Create a square image
    m = floor(sqrt(N));

    if m < 2
        m = 2;
    end

    % Crop or pad to get m^2 samples
    if N >= m^2
        X_crop = X_norm(1:m^2);
    else
        X_crop = [X_norm; zeros(m^2 - N, 1)];
    end

    % Reshape into a 2D matrix
    img_gray = reshape(X_crop, m, m);
end

function img_ssm = compute_self_similarity_matrix(X_norm)
    % Self-similarity matrix: distances |x_i - x_j|

    N = length(X_norm);

    % Limit size to avoid very large matrices
    maxN = 256;
    if N > maxN
        idx = round(linspace(1, N, maxN));
        X_ssm = X_norm(idx);
    else
        X_ssm = X_norm;
    end

    M = length(X_ssm);

    % Compute distance matrix
    img_ssm = zeros(M, M);
    for i = 1:M
        img_ssm(i, :) = abs(X_ssm(i) - X_ssm(:)).';
    end
end

%% ========================================================================
%% HELPER FUNCTION: RESIZE IMAGE
%% ========================================================================

function img_resized = resize_to_target(img, target_size)
    % Resizes an image to the target size using interpolation

    [rows, cols] = size(img);

    if rows == target_size && cols == target_size
        img_resized = img;
        return;
    end

    try
        % Use imresize if available
        img_resized = imresize(img, [target_size, target_size], 'bilinear');
    catch
        % Simple manual implementation if imresize is not available
        row_idx = round(linspace(1, rows, target_size));
        col_idx = round(linspace(1, cols, target_size));
        img_resized = img(row_idx, col_idx);
    end
end
