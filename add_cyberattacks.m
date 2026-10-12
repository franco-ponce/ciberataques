function [X_attacked, attack_labels] = add_cyberattacks(X, varargin)
% ADD_CYBERATTACKS Adds different types of cyberattacks and distortions to time series
%
% SYNTAX:
%   [X_attacked, attack_labels] = add_cyberattacks(X, 'param1', value1, ...)
%
% INPUTS:
%   X               - Data matrix [N x M] where N=samples, M=variables
%
% OPTIONAL PARAMETERS (name-value pairs):
%   'AttackType'    - Attack type: 'drift', 'noise', 'packet_loss',
%                     'retained_value', 'periodic_dropout', 'FDI', 'DoS',
%                     'replay', 'all' (default: 'drift')
%   'Columns'       - Indices of the columns to attack (default: all)
%   'TimeStart'     - Attack start time (default: round(N*0.3))
%   'TimeEnd'       - Attack end time (default: round(N*0.5))
%   'Intensity'     - Attack intensity (default: 0.5)
%   'Probability'   - Probability for stochastic attacks (default: 0.5)
%   'UpdateInterval'- Update interval for retained value (default: 60)
%   'DropPeriod'    - Period for periodic dropout (default: 4)
%   'ReplayWindow'  - Window size for replay attack (default: 64)
%   'ReplayReps'    - Repetitions for replay attack (default: 5)
%   'FDI_Amplitude' - Amplitude for FDI [min, max] (default: [1.7, 6.4])
%   'Verbose'       - Show information (true/false, default: true)
%
% OUTPUTS:
%   X_attacked      - Matrix with attacked data [N x M]
%   attack_labels   - Binary vector [N x 1]: 1=attacked, 0=normal
%
% ATTACK TYPES:
%   'drift'           - Gradual sensor drift
%   'noise'           - Additional Gaussian noise
%   'packet_loss'     - Random packet loss
%   'retained_value'  - Value held and periodically updated
%   'periodic_dropout'- Periodic packet dropout
%   'FDI'             - False Data Injection
%   'DoS'             - Denial of Service
%   'replay'          - Data replay attack
%   'all'             - Applies multiple attacks over different intervals
%
% EXAMPLES:
%   % Example 1: Drift attack on columns 1-3
%   [X_new, labels] = add_cyberattacks(X, 'AttackType', 'drift', ...
%                                      'Columns', 1:3, 'Intensity', 5);
%
%   % Example 2: FDI attack with custom parameters
%   [X_new, labels] = add_cyberattacks(X, 'AttackType', 'FDI', ...
%                                      'Columns', [7,8], ...
%                                      'TimeStart', 1000, ...
%                                      'TimeEnd', 1500, ...
%                                      'Probability', 0.5, ...
%                                      'FDI_Amplitude', [2, 8]);
%
%   % Example 3: Multiple attacks
%   [X_new, labels] = add_cyberattacks(X, 'AttackType', 'all');
%
% AUTHOR: Claude
% DATE: 2025

%% Input validation
% Without data there is nothing to attack
if nargin < 1
    error('add_cyberattacks:noInput', 'Se requiere al menos la matriz X de entrada.');
end

[N, M] = size(X);  % N = number of samples (rows), M = number of variables (columns)

%% Argument parser
% Defines each optional parameter with its default value and its validation
p = inputParser;
addRequired(p, 'X', @isnumeric);
addParameter(p, 'AttackType', 'drift', @ischar);
addParameter(p, 'Columns', 1:M, @isnumeric);
addParameter(p, 'TimeStart', round(N*0.3), @isnumeric);     % start at 30% of the data
addParameter(p, 'TimeEnd', round(N*0.5), @isnumeric);       % end at 50% of the data
addParameter(p, 'Intensity', 0.5, @isnumeric);
addParameter(p, 'Probability', 0.5, @(x) x>=0 && x<=1);    % must be in [0, 1]
addParameter(p, 'UpdateInterval', 60, @isnumeric);
addParameter(p, 'DropPeriod', 4, @isnumeric);
addParameter(p, 'ReplayWindow', 64, @isnumeric);
addParameter(p, 'ReplayReps', 5, @isnumeric);
addParameter(p, 'FDI_Amplitude', [1.7, 6.4], @isnumeric);
addParameter(p, 'Verbose', true, @islogical);

parse(p, X, varargin{:});  % Reads the name-value pairs passed by the user

% Extract parameters
attack_type = lower(p.Results.AttackType);  % lowercase: 'FDI' and 'fdi' are equivalent
cols = p.Results.Columns;
t0 = p.Results.TimeStart;
t1 = p.Results.TimeEnd;
intensity = p.Results.Intensity;
prob = p.Results.Probability;
update_int = p.Results.UpdateInterval;
drop_period = p.Results.DropPeriod;
replay_win = p.Results.ReplayWindow;
replay_reps = p.Results.ReplayReps;
fdi_amp = p.Results.FDI_Amplitude;
verbose = p.Results.Verbose;

% Validations
% The attack interval must have a positive length
if t1 <= t0
    error('add_cyberattacks:invalidTime', 'TimeEnd debe ser mayor que TimeStart.');
end
% If the end exceeds the data length, it is clipped to N
if t1 > N
    t1 = N;
    if verbose
        warning('TimeEnd ajustado al tamaño de los datos: %d', N);
    end
end
% The requested columns must exist in X
if any(cols > M) || any(cols < 1)
    error('add_cyberattacks:invalidColumns', 'Índices de columnas fuera de rango.');
end

%% Initialization
X_attacked = X;  % Copy of the data
attack_labels = zeros(N, 1);  % Label vector
                              % (0 = normal; each attack sets 1 in its interval)

%% Apply attack by type
% Each case calls its helper function (defined below)
switch attack_type
    case 'drift'
        % Adds an increasing ramp -> sensor losing calibration
        [X_attacked, attack_labels] = apply_drift(X_attacked, attack_labels, ...
            t0, t1, cols, intensity, verbose);

    case 'noise'
        % Adds Gaussian noise proportional to the standard deviation
        [X_attacked, attack_labels] = apply_noise(X_attacked, attack_labels, ...
            t0, t1, cols, intensity, verbose);

    case 'packet_loss'
        % Sets random samples to zero (lost packets)
        [X_attacked, attack_labels] = apply_packet_loss(X_attacked, attack_labels, ...
            t0, t1, cols, prob, verbose);

    case 'retained_value'
        % Freezes the value and only updates it every 'update_int' samples
        [X_attacked, attack_labels] = apply_retained_value(X_attacked, attack_labels, ...
            t0, t1, cols, update_int, verbose);

    case 'periodic_dropout'
        % Sets one sample to zero every 'drop_period' samples
        [X_attacked, attack_labels] = apply_periodic_dropout(X_attacked, attack_labels, ...
            t0, t1, cols, drop_period, verbose);

    case 'fdi'
        % Injects false spikes (+/-) at random samples
        [X_attacked, attack_labels] = apply_fdi(X_attacked, attack_labels, ...
            t0, t1, cols, prob, fdi_amp, verbose);

    case 'dos'
        % Blocks transmissions: the last received value is repeated
        [X_attacked, attack_labels] = apply_dos(X_attacked, attack_labels, ...
            t0, t1, cols, prob, verbose);

    case 'replay'
        % Records a segment and plays it back several times (ignores TimeEnd)
        [X_attacked, attack_labels] = apply_replay(X_attacked, attack_labels, ...
            t0, replay_win, replay_reps, cols, verbose);

    case 'all'
        % Demo scenario with several attacks over fixed segments
        % (ignores Columns, TimeStart, TimeEnd, etc.)
        [X_attacked, attack_labels] = apply_multiple_attacks(X_attacked, N, M, verbose);

    otherwise
        error('add_cyberattacks:unknownType', ...
            'Tipo de ataque desconocido: %s', attack_type);
end

end

%% ========================================================================
%                           HELPER FUNCTIONS
%  ========================================================================
% All of them follow the same pattern:
%   1) copy X, 2) modify columns 'cols' in [t0, t1],
%   3) set labels = 1 in that interval, 4) print a summary if verbose.

function [X_out, labels] = apply_drift(X, labels, t0, t1, cols, max_drift, verbose)
    % Gradual drift attack
    % The deviation grows linearly from 0 to max_drift within the interval
    X_out = X;
    L = t1 - t0 + 1;                     % number of attacked samples
    drift = linspace(0, max_drift, L)';  % ramp 0 -> max_drift (column vector)

    for c = cols
        X_out(t0:t1, c) = X_out(t0:t1, c) + drift;  % adds the ramp to the signal
    end

    labels(t0:t1) = 1;  % marks the interval as attacked

    if verbose
        fprintf('✓ Ataque DRIFT aplicado: t=[%d, %d], columnas=%s, max_drift=%.2f\n', ...
            t0, t1, mat2str(cols), max_drift);
    end
end

function [X_out, labels] = apply_noise(X, labels, t0, t1, cols, intensity, verbose)
    % Gaussian noise attack
    % Noise amplitude = intensity * column std (scales with each variable)
    X_out = X;

    for c = cols
        epsilon = intensity * std(X(:, c));          % noise standard deviation
        noise = epsilon * randn(t1-t0+1, 1);         % noise N(0, epsilon^2)
        X_out(t0:t1, c) = X_out(t0:t1, c) + noise;   % added to the signal
    end

    labels(t0:t1) = 1;

    if verbose
        fprintf('✓ Ataque NOISE aplicado: t=[%d, %d], columnas=%s, intensidad=%.2f\n', ...
            t0, t1, mat2str(cols), intensity);
    end
end

function [X_out, labels] = apply_packet_loss(X, labels, t0, t1, cols, prob, verbose)
    % Random packet loss attack
    % Each sample is lost (set to 0) with probability 'prob'
    X_out = X;
    L = t1 - t0 + 1;

    for c = cols
        loss_vector = rand(L, 1);              % uniform numbers in [0, 1)
        loss_index = loss_vector >= prob;  % 1=keep, 0=lose
        X_out(t0:t1, c) = X_out(t0:t1, c) .* loss_index;  % multiplying by 0 = lost
    end

    labels(t0:t1) = 1;

    if verbose
        fprintf('✓ Ataque PACKET_LOSS aplicado: t=[%d, %d], columnas=%s, prob=%.2f\n', ...
            t0, t1, mat2str(cols), prob);
    end
end

function [X_out, labels] = apply_retained_value(X, labels, t0, t1, cols, update_int, verbose)
    % Retained value attack with periodic update
    % The signal becomes "stepped": it is only refreshed every update_int samples
    X_out = X;

    for c = cols
        retained = X_out(t0, c);  % first retained value
        for i = t0:t1
            if mod(i - t0, update_int) == 0
                retained = X_out(i, c);  % time to update: take the real value
            end
            X_out(i, c) = retained;      % otherwise repeat the retained value
        end
    end

    labels(t0:t1) = 1;

    if verbose
        fprintf('✓ Ataque RETAINED_VALUE aplicado: t=[%d, %d], columnas=%s, intervalo=%d\n', ...
            t0, t1, mat2str(cols), update_int);
    end
end

function [X_out, labels] = apply_periodic_dropout(X, labels, t0, t1, cols, period, verbose)
    % Periodic packet dropout attack
    % Unlike packet_loss, the drops here are regular, not random
    X_out = X;

    for c = cols
        for i = t0:t1
            if mod(i - t0, period) == 0
                X_out(i, c) = 0;  % one sample lost every 'period' samples
            end
        end
    end

    labels(t0:t1) = 1;

    if verbose
        fprintf('✓ Ataque PERIODIC_DROPOUT aplicado: t=[%d, %d], columnas=%s, período=%d\n', ...
            t0, t1, mat2str(cols), period);
    end
end

function [X_out, labels] = apply_fdi(X, labels, t0, t1, cols, prob, amp_range, verbose)
    % FDI attack (False Data Injection)
    % At each sample, with probability 'prob', a spike of random
    % magnitude in [amp_min, amp_max] and random sign is added
    X_out = X;
    L = t1 - t0 + 1;

    for c = cols
        % Generate random injections
        inject_mask = rand(L, 1) < prob;  % 1 = this sample gets an injection
        inject_values = amp_range(1) + (amp_range(2) - amp_range(1)) * rand(L, 1);  % uniform magnitude within the range
        inject_signs = 2 * (rand(L, 1) > 0.5) - 1;  % ±1

        % Only added where inject_mask = 1
        X_out(t0:t1, c) = X_out(t0:t1, c) + inject_mask .* inject_values .* inject_signs;
    end

    labels(t0:t1) = 1;

    if verbose
        fprintf('✓ Ataque FDI aplicado: t=[%d, %d], columnas=%s, prob=%.2f, amp=[%.1f,%.1f]\n', ...
            t0, t1, mat2str(cols), prob, amp_range(1), amp_range(2));
    end
end

function [X_out, labels] = apply_dos(X, labels, t0, t1, cols, prob, verbose)
    % DoS attack (Denial of Service)
    % NOTE: here 'prob' is the probability that the data DOES arrive
    % (low prob = stronger attack)
    X_out = X;
    L = t1 - t0 + 1;

    for c = cols
        % Probability of successful transmission
        transmit_mask = rand(L, 1) < prob;  % 1 = arrives, 0 = blocked

        % Hold the last valid value when there is no transmission
        retained = X_out(t0, c);
        for i = 1:L
            if transmit_mask(i)
                retained = X_out(t0+i-1, c);  % data received: update
            else
                X_out(t0+i-1, c) = retained;  % data blocked: repeat the last one
            end
        end
    end

    labels(t0:t1) = 1;

    if verbose
        fprintf('✓ Ataque DoS aplicado: t=[%d, %d], columnas=%s, prob_transmisión=%.2f\n', ...
            t0, t1, mat2str(cols), prob);
    end
end

function [X_out, labels] = apply_replay(X, labels, t0, win_size, reps, cols, verbose)
    % Replay attack
    % Records [t0, t0+win_size-1] and plays it back 'reps' times right after
    X_out = X;

    % Capture data window
    t_end_capture = t0 + win_size - 1;
    captured_data = X(t0:t_end_capture, cols);  % segment "recorded" by the attacker

    % Repeat the captured window
    t_replay_start = t_end_capture + 1;
    t_replay_end = min(t_replay_start + win_size * reps - 1, size(X, 1));  % do not exceed N

    replay_length = t_replay_end - t_replay_start + 1;
    replayed_data = repmat(captured_data, ceil(replay_length / win_size), 1);  % stacks copies
    replayed_data = replayed_data(1:replay_length, :);  % trims to the exact length

    X_out(t_replay_start:t_replay_end, cols) = replayed_data;  % replaces (does not add)
    labels(t_replay_start:t_replay_end) = 1;  % only the playback is labeled as attack

    if verbose
        fprintf('✓ Ataque REPLAY aplicado: t_captura=[%d, %d], t_replay=[%d, %d], columnas=%s\n', ...
            t0, t_end_capture, t_replay_start, t_replay_end, mat2str(cols));
    end
end

function [X_out, labels] = apply_multiple_attacks(X, N, M, verbose)
    % Apply multiple attacks over different intervals
    % The attacks follow one another, separated by normal segments
    X_out = X;
    labels = zeros(N, 1);

    % Define time segments for the different attacks
    seg_size = floor(N / 6);  % base time unit used to place each attack

    % Attack 1: Drift
    % (columns 1-2, max drift = 5)
    t0_1 = seg_size;
    t1_1 = t0_1 + round(seg_size * 0.6);
    [X_out, labels] = apply_drift(X_out, labels, t0_1, t1_1, 1:min(2,M), 5, verbose);

    % Attack 2: Noise
    % (columns 1-3, intensity 0.5; starts after a normal gap of 0.2*seg_size)
    t0_2 = t1_1 + round(seg_size * 0.2);
    t1_2 = t0_2 + round(seg_size * 0.6);
    [X_out, labels] = apply_noise(X_out, labels, t0_2, t1_2, 1:min(3,M), 0.5, verbose);

    % Attack 3: Packet Loss
    % (columns 2-3, 50% loss)
    t0_3 = t1_2 + round(seg_size * 0.2);
    t1_3 = t0_3 + round(seg_size * 0.6);
    [X_out, labels] = apply_packet_loss(X_out, labels, t0_3, t1_3, min(2,M):min(3,M), 0.5, verbose);

    % Attack 4: Retained Value
    % (columns 1-2, updates every 90 samples)
    t0_4 = t1_3 + round(seg_size * 0.2);
    t1_4 = t0_4 + round(seg_size * 0.8);
    [X_out, labels] = apply_retained_value(X_out, labels, t0_4, t1_4, 1:min(2,M), 90, verbose);

    % Attack 5: FDI
    % Only if there are at least 7 variables (attacks columns 7-8)
    if M >= 7
        t0_5 = t1_4 + round(seg_size * 0.3);
        t1_5 = t0_5 + round(seg_size * 0.6);
        [X_out, labels] = apply_fdi(X_out, labels, t0_5, t1_5, 7:min(8,M), 0.5, [1.7, 6.4], verbose);
    end

    % Summary: how many samples were labeled as attack
    if verbose
        fprintf('\n✓ Aplicación de múltiples ataques completada.\n');
        fprintf('  Total de muestras atacadas: %d / %d (%.1f%%)\n', ...
            sum(labels), N, 100*sum(labels)/N);
    end
end
