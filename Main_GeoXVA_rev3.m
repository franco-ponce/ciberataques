% Clean workspace and load the boiler dataset
close all
clear all
clc

load("boiler_data_set.mat")  % loads table 'boilerdata'

%% ========================================================================
%% CONFIGURATION
%% ========================================================================
boilerdata2=table2timetable(boilerdata(149781:263700,:));  % select time range and convert to timetable
boilerdata2=retime(boilerdata2, 'minutely', 'linear');  % resample to 1-minute steps (linear interpolation)
boilerdata2 = removevars(boilerdata2, {'PrPsteamPcr','Ln_Temp_steam'});  % drop unused variables
vars = boilerdata2.Properties.VariableNames;  % names of the remaining variables

% Replace negative readings (sensor errors) with the previous value
for v = 1:numel(vars)
    x = boilerdata2.(vars{v});   % extract the column
    for i = 2:length(x)
        if x(i) < 0
            x(i) = x(i-1);  % replace with the previous positive value
        end
    end
    boilerdata2.(vars{v}) = x;   % store the corrected column
end

L=8;   %% time window size
Nwindows_tr=2500;  % number of training windows
Nwindows_test=1250;  % number of test windows
%N=L*Nwindows;
X1=boilerdata2.Variables;  % numeric matrix [samples x variables]
fs = 1;             % Sampling frequency (Hz)


%% add manual attacks + distortions
%% attack 1: drift
[X1_attacked, attack_labels1] = add_cyberattacks(X1, 'AttackType', 'drift', 'Columns', 1,'TimeStart', 20000, ...
                                     'TimeEnd', 22000, 'Intensity', 0.075*max(X1(:,1)));
figure, plot(X1(:,1)),hold on, plot(X1_attacked(:,1))  % original vs attacked signal

%% attack 2: FDI with custom parameters
[X_new, attack_labels2] = add_cyberattacks(X1_attacked, 'AttackType', 'FDI', ...
                                     'Columns', [7,8], ...
                                     'TimeStart', 23000, ...
                                     'TimeEnd', 25000, ...
                                     'Probability', 0.25, ...
                                     'FDI_Amplitude', [0.05*max(X1(:,7)), 0.05*max(X1(:,8))]);
% FDI effect on variables 7 and 8
figure, plot(X_new(:,7)),hold on, plot(X1_attacked(:,7))
figure, plot(X_new(:,8)),hold on, plot(X1_attacked(:,8))

%% attack 3: DoS
% NOTE: uses 'packet_loss' (zeros), not 'dos' (holds last value)
[X_boiler, attack_labels3] = add_cyberattacks(X_new, 'AttackType', 'packet_loss', ...
                                     'Columns', 5, ...
                                     'TimeStart', 26000, ...
                                     'TimeEnd', 27500, ...
                                     'Probability', 0.75);
figure, plot(X_new(:,5)),hold on, plot(X_boiler(:,5))  % before vs after packet loss

ref_state=attack_labels1+attack_labels2+attack_labels3;  % combine labels of the 3 attacks
ref_state = max(0, min(ref_state, 1));  % clip to binary 0/1 (attacks may overlap)
figure, plot(ref_state);  % reference attack labels

%% Build train and test datasets
Xtrain=X_boiler(1:Nwindows_tr*L,:);  % train: first Nwindows_tr*L samples (before the attacks)
Xtest=X_boiler(20000:20000+Nwindows_test*L-1,:);  % test: starts at sample 20000 (covers the attacks)
ref_state_test=ref_state(20000:20000+Nwindows_test*L-1);  % ground-truth labels of the test segment

% Training image tensor: each window of L samples -> 24 images per variable
jj=1;
for j=L:L:size(Xtrain,1)  % end index of each non-overlapping window
    for l=1:size(Xtrain,2)  % loop over variables
        temp=Xtrain(j-L+1:j,l);  % current window of variable l
        [T1, dT1, ddT1] = timeseries_to_images_complete(temp,1);  % 8 encodings of signal, 1st and 2nd derivative
        TT= cat(3, T1, dT1, ddT1);  % stack into 24 channels
        m=size(TT,3);
        m1=m*(l-1)+1;  % first channel of variable l
        m2=m*l;  % last channel of variable l
        XIMtr(:,:,m1:m2,jj)=TT;  % XIMtr: [H x W x (24*nVars) x nWindows]
    end
    jj=jj+1;  % next window
end


% Test image tensor (same procedure) + window labels
jj=1;
for j=L:L:size(Xtest,1)
    for l=1:size(Xtest,2)
        temp=Xtest(j-L+1:j,l);
        [T1, dT1, ddT1] = timeseries_to_images_complete(temp,1);
        TT= cat(3, T1, dT1, ddT1);
        m=size(TT,3);
        m1=m*(l-1)+1;
        m2=m*l;
        XIMtest(:,:,m1:m2,jj)=TT;
    end
    target(jj)=sum(ref_state_test(j-L+1:j));  % number of attacked samples in the window
    jj=jj+1;
end

target = max(0, min(target, 1));  % window = attack (1) if any sample is attacked

% UNIFIED_GEOMAE_COMPREHENSIVE.m
% =========================================================================
% Comprehensive script combining:
%   - 13 Geometric Autoencoder variants
%   - Physics-Informed Neural Networks (optional via flag)
%   - Process Monitoring: PCA, PLS, KPCA, KPLS
%   - Complete KPI computation (performance + computational cost)
%   - Cross-platform resource monitoring (Windows/Linux)
%   - Live visual monitoring during training
%   - Geometric performance analysis
%   - Explainability: SHAP (local/global) + LIME
%   - Monte Carlo classification analysis with statistics
% =========================================================================
% Author: Comprehensive Integration
% Date: 2025
% Requirements: MATLAB R2021b+, Deep Learning Toolbox, Statistics Toolbox
% =========================================================================

% clear; close all; clc;
% rng(42); % Reproducibility

%% ========================================================================
% SECTION 1: CONFIGURATION
% =========================================================================
fprintf('=== UNIFIED GEOMETRIC AUTOENCODER FRAMEWORK ===\n\n');

% --- Flags ---
CONFIG = struct();
CONFIG.USE_PHYSICS_INFORMED = false;                % Enable PINN heads for physics
CONFIG.USE_PROCESS_MONITORING = true;               % Enable PCA/PLS/KPCA/KPLS
CONFIG.USE_EXPLAINABILITY = false;                  % Enable SHAP/LIME analysis
CONFIG.USE_MONTE_CARLO = false;                     % Enable MC classification analysis
CONFIG.MONTE_CARLO_RUNS = 5;                        % Number of MC trials
CONFIG.VERBOSE_PLOTS = true;                        % Show training plots
CONFIG.SAVE_RESULTS = true;                         % Save intermediate results

% --- Directories ---
if ~exist('results','dir'), mkdir('results'); end
if ~exist('results/figures','dir'), mkdir('results/figures'); end
if ~exist('results/monitoring','dir'), mkdir('results/monitoring'); end
if ~exist('results/explainability','dir'), mkdir('results/explainability'); end
if ~exist('results/monte_carlo','dir'), mkdir('results/monte_carlo'); end

% --- Training Hyperparameters ---
TRAIN = struct();
TRAIN.numEpochs = 5000;  % max number of epochs
TRAIN.miniBatchSize = 64;  % samples per minibatch
TRAIN.baseLR = 5e-4;  % base learning rate
TRAIN.momentum = 0.95;  % SGD momentum
TRAIN.decayRate = 0.98;  % LR decay factor
TRAIN.decaySteps = 100;  % apply decay every N iterations
TRAIN.patience = 50;  % early stopping: iterations without improvement
TRAIN.minIters = 500;  % min iterations before early stopping
TRAIN.tolImprove = 1e-6;  % min loss improvement
TRAIN.minLR = 1e-6;  % lower bound for LR

% --- Loss Weights (Physics-Informed preset) ---
% rec=reconstruction, l2=weights, norm=latent norm, geo=geometric, phys=PINN, var=temporal, kl=VAE
if CONFIG.USE_PHYSICS_INFORMED
    WEIGHTS = struct('rec',1.0, 'l2',1e-6, 'norm',5e-4, ...
                     'geo',0.20, 'phys',0.15, 'var',5e-5, 'kl',1e-3);
else
    WEIGHTS = struct('rec',1.0, 'l2',1e-6, 'norm',5e-4, ...
                     'geo',0.25, 'phys',0.0, 'var',5e-5, 'kl',1e-3);
end

% --- Autoencoder Variants (13 total) ---
% VARIANTS = {'POINCARE', 'LORENTZ', 'KLEIN','EUCLIDEAN',  ...
%             'SPHERE', 'SPHERE_R', 'PRODUCT_EHS', 'H2_PRODUCT', ...
%             'ELLIPTIC', 'GEODESIC_TANGENT', 'VAE', 'CONV_AE', 'ROBUST_AE'};
% Only POINCARE is active; uncomment to train more variants
VARIANTS = {'POINCARE'}%, 'LORENTZ', 'KLEIN','EUCLIDEAN',  ...
            %'GEODESIC_TANGENT', 'VAE', 'CONV_AE', 'ROBUST_AE'};


% Print configuration summary
fprintf('Configuration:\n');
fprintf('  Physics-Informed: %s\n', mat2str(CONFIG.USE_PHYSICS_INFORMED));
fprintf('  Process Monitoring: %s\n', mat2str(CONFIG.USE_PROCESS_MONITORING));
fprintf('  Explainability: %s\n', mat2str(CONFIG.USE_EXPLAINABILITY));
fprintf('  Monte Carlo Runs: %d\n', CONFIG.MONTE_CARLO_RUNS);
fprintf('  Variants: %d\n\n', numel(VARIANTS));

%% ========================================================================
% SECTION 2: DATA LOADING
% =========================================================================
fprintf('Loading dataset...\n');
% Autoencoder: input and target are the same images
XTrain=XIMtr;
YTrain=XIMtr;
XTest=XIMtest;
YTest=XIMtest;

%[XTrain, XTest, YTrain, YTest, imageSize] = load_dataset();
[H, W, C, NTrain] = size(XTrain);  % height, width, channels, samples
imageSize=[H,W,C];
NTest = size(XTest, 4);
fprintf('  Train samples: %d, Test samples: %d\n', NTrain, NTest);
fprintf('  Image size: %dx%dx%d\n\n', H, W, C);

% Physics labels (if using PINN)
if CONFIG.USE_PHYSICS_INFORMED
    eta_dir_train = YTrain(:,1);
    eta_ind_train = YTrain(:,2);
    eta_dir_test  = YTest(:,1);
    eta_ind_test  = YTest(:,2);
else
    eta_dir_train = []; eta_ind_train = [];
    eta_dir_test  = []; eta_ind_test  = [];
end

%% ========================================================================
% SECTION 3: INITIALIZE RESOURCE MONITORING
% =========================================================================
fprintf('Initializing resource monitoring...\n');
powerModel = struct('idle_w',30, 'cpu_tdp_w',65, 'gpu_tdp_w',150, 'other_w',10);  % power model (W) used to estimate energy
sysMonitor = initSystemMonitoring(powerModel);

%% ========================================================================
% SECTION 4: MAIN TRAINING LOOP (ALL VARIANTS)
% =========================================================================
results = struct();  % stores everything per variant
allGeometricKPIs = table();  % summary KPI table of all variants

for v = 1:numel(VARIANTS)
    variantName = VARIANTS{v};
    fprintf('\n========================================\n');
    fprintf('VARIANT %d/%d: %s\n', v, numel(VARIANTS), variantName);
    fprintf('========================================\n');

    tVariantStart = tic;  % timer for this variant

    % Get variant specification
    spec = getVariantSpec(variantName);
    latentDim = spec.latentDim;  % latent space dimension
    hiddenSize = spec.hiddenSize;  % number of conv filters
    learnRate = spec.learnRate * TRAIN.baseLR;  % effective learning rate

    % Adjust weights if variant has overrides
    weights_eff = WEIGHTS;
    if isfield(spec, 'weight_override')
        fn = fieldnames(spec.weight_override);
        for k = 1:numel(fn)
            weights_eff.(fn{k}) = spec.weight_override.(fn{k});
        end
    end
    % Weight vector in the order used by computeLoss
    beta = [weights_eff.rec, weights_eff.l2, weights_eff.norm, ...
            weights_eff.geo, weights_eff.phys, weights_eff.var, weights_eff.kl];

    Rparam = dlarray(spec.R, 'CB');  % trainable curvature/radius parameter R

    % Build networks
    fprintf('  Building encoder/decoder...\n');
    [netE, netD, encShape] = buildAutoencoder(imageSize, latentDim, hiddenSize, spec);

    if CONFIG.USE_PHYSICS_INFORMED
        fprintf('  Building physics heads...\n');
        physInDim = getPhysicsInputDim(spec);
        [netDir, netInd] = buildPhysicsHeads(physInDim);
    else
        netDir = []; netInd = [];
    end

    % Initialize monitoring panel
    if CONFIG.VERBOSE_PLOTS
        monPanel = createMonitoringPanel(variantName, imageSize);
    else
        monPanel = [];
    end

    % Training state
    velE = []; velD = []; velR = []; velDir = []; velInd = [];  % SGDM momentum buffers
    bestLoss = inf; waitCount = 0; iteration = 0;  % early stopping state
    bestIter = 0; bestTime = 0;
    trace = struct('loss',[],'recon',[],'geo',[],'phys',[],'var',[],'kl',[],'R',[]);
    prev = struct('E',[],'D',[],'Dir',[],'Ind',[]);

    % Metrics logging
    metricsLog = table();

    % Prepare minibatch queue
    mbq = prepareMinibatchQueue(XTrain, eta_dir_train, eta_ind_train, ...
                                 TRAIN.miniBatchSize, CONFIG.USE_PHYSICS_INFORMED);

    % Training loop
    fprintf('  Training...\n');
    tTrainStart = tic;

    for epoch = 1:TRAIN.numEpochs
        reset(mbq);  % restart queue each epoch
        shuffle(mbq);  % shuffle training data

        while hasdata(mbq)
            iteration = iteration + 1;  % global iteration counter

            % Learning rate decay
            if mod(iteration, TRAIN.decaySteps) == 0
                learnRate = max(TRAIN.minLR, learnRate * TRAIN.decayRate);
            end

            % Get minibatch
            if CONFIG.USE_PHYSICS_INFORMED
                [Xb, ydirb, yindb] = next(mbq);
            else
                Xb = next(mbq);
                ydirb = []; yindb = [];
            end

            % Forward + backward pass
            [L, grads, parts] = dlfeval(@computeLoss, Xb, ydirb, yindb, ...
                netE, netD, netDir, netInd, Rparam, beta, spec, prev, ...
                CONFIG.USE_PHYSICS_INFORMED);

            % Check for NaN/Inf
            Lval = double(gather(extractdata(L)));  % loss as double scalar
            if ~isfinite(Lval)
                learnRate = max(TRAIN.minLR, 0.5*learnRate);  % halve LR and skip this batch
                continue;
            end

            % Update parameters
            [netE.Learnables, velE] = sgdmupdate(netE.Learnables, grads.gE, velE, learnRate, TRAIN.momentum);
            [netD.Learnables, velD] = sgdmupdate(netD.Learnables, grads.gD, velD, learnRate, TRAIN.momentum);
            if CONFIG.USE_PHYSICS_INFORMED
                [netDir.Learnables, velDir] = sgdmupdate(netDir.Learnables, grads.gDir, velDir, learnRate, TRAIN.momentum);
                [netInd.Learnables, velInd] = sgdmupdate(netInd.Learnables, grads.gInd, velInd, learnRate, TRAIN.momentum);
            end
            [Rparam, velR] = sgdmupdate(Rparam, grads.gR, velR, learnRate, TRAIN.momentum);
            Rparam = clampR(Rparam, spec);  % keep R within its valid range

            % Record traces
            trace.loss(end+1) = Lval;
            trace.recon(end+1) = double(gather(extractdata(parts.Lrec)));
            trace.geo(end+1) = double(gather(extractdata(parts.Lgeo)));
            trace.phys(end+1) = double(gather(extractdata(parts.Lphys)));
            trace.var(end+1) = double(gather(extractdata(parts.Lvar)));
            trace.kl(end+1) = double(gather(extractdata(parts.Lkl)));
            trace.R(end+1) = double(gather(extractdata(Rparam)));

            % Sample system metrics
            [mRow, sysMonitor] = sampleSystemMetrics(sysMonitor, powerModel);
            newRow = struct2table(struct( ...
                'variant', string(variantName), ...
                'iter', iteration, ...
                'epoch', epoch, ...
                'elapsed_s', toc(tTrainStart), ...
                'cpu_pct', mRow.cpu_pct, ...
                'ram_used_mb', mRow.ram_used_mb, ...
                'ram_pct', mRow.ram_pct, ...
                'gpu_util_pct', mRow.gpu_util_pct, ...
                'gpu_mem_used_mb', mRow.gpu_mem_used_mb, ...
                'gpu_power_w', mRow.gpu_power_w, ...
                'cpu_power_w', mRow.cpu_power_w, ...
                'total_power_w', mRow.total_power_w, ...
                'cum_energy_j', mRow.cum_energy_j, ...
                'lr', learnRate, ...
                'loss', Lval, ...
                'loss_rec', trace.recon(end), ...
                'loss_geo', trace.geo(end), ...
                'loss_phys', trace.phys(end), ...
                'loss_var', trace.var(end), ...
                'loss_kl', trace.kl(end) ...
            ));
            metricsLog = [metricsLog; newRow]; %#ok<AGROW>

            % Update monitoring panel
            if CONFIG.VERBOSE_PLOTS && ~isempty(monPanel) && mod(iteration, 10) == 0
                updateMonitoringPanel(monPanel, iteration, metricsLog, trace, Rparam);
            end

            % Early stopping
            if Lval < (bestLoss - TRAIN.tolImprove)  % improved: save best
                bestLoss = Lval;
                bestIter = iteration;
                bestTime = newRow.elapsed_s;
                waitCount = 0;
            else
                waitCount = waitCount + 1;  % no improvement
            end

            % Stop when patience runs out (after minIters)
            if (iteration >= TRAIN.minIters) && (waitCount >= TRAIN.patience)
                break;
            end

            prev = snapshotNetworks(netE, netD, netDir, netInd);  % saved for the temporal variability loss
        end

        % Propagate the break to the epoch loop
        if (iteration >= TRAIN.minIters) && (waitCount >= TRAIN.patience)
            break;
        end
    end

    tTrainElapsed = toc(tTrainStart);  % total training time
    fprintf('  Training complete: %d iters, %.2fs, best loss=%.6f\n', ...
            iteration, tTrainElapsed, bestLoss);
    
    % Evaluate on test set
    fprintf('  Evaluating on test set...\n');
    [testResults, geoKPIs] = evaluateModel(XTest, eta_dir_test, eta_ind_test, ...
        netE, netD, netDir, netInd, Rparam, spec, CONFIG.USE_PHYSICS_INFORMED);
    
    % Compute geometric performance KPIs
    if isGeometricVariant(variantName)
        fprintf('  Computing geometric KPIs...\n');
        geoAnalysis = analyzeGeometricPerformance(testResults.Z, testResults.Xhat, ...
                                                   XTest, spec, Rparam);
        geoKPIs = mergeStructs(geoKPIs, geoAnalysis);
    end
    
    % Store results
    results.(variantName) = struct();
    results.(variantName).netE = netE;
    results.(variantName).netD = netD;
    results.(variantName).netDir = netDir;
    results.(variantName).netInd = netInd;
    results.(variantName).Rparam = Rparam;
    results.(variantName).spec = spec;
    results.(variantName).trace = trace;
    results.(variantName).metricsLog = metricsLog;
    results.(variantName).testResults = testResults;
    results.(variantName).geoKPIs = geoKPIs;
    results.(variantName).bestLoss = bestLoss;
    results.(variantName).bestIter = bestIter;
    results.(variantName).bestTime = bestTime;
    results.(variantName).totalTime = toc(tVariantStart);
    
    % Add to consolidated KPI table
    kpiRow = struct2table(struct( ...
        'variant', string(variantName), ...
        'best_loss', bestLoss, ...
        'train_time_s', tTrainElapsed, ...
        'total_energy_j', metricsLog.cum_energy_j(end), ...
        'test_mse', testResults.MSE, ...
        'test_mae', testResults.MAE, ...
        'test_ssim', testResults.SSIM, ...
        'geo_dist_mean', geoKPIs.geo_dist_mean, ...
        'geo_dist_std', geoKPIs.geo_dist_std ...
    ));
    allGeometricKPIs = [allGeometricKPIs; kpiRow];  % append row
    
    fprintf('  Variant %s complete.\n', variantName);
    % Excel file name (can be changed)
    excelFile = ['resultados_summary_',string(variantName),'.xlsx'];

    % Sheet name = variant name
    sheetName = char(string(variantName));   % in case it comes as a string

    % Save this variant's KPI row in its own sheet
    % (overwrites the sheet if it exists; otherwise creates it)
    writetable(kpiRow, excelFile,'Sheet', sheetName,'WriteMode', 'overwrite');
end

fprintf('\n=== All variants trained ===\n\n');


%% ========================================================================
% PROCESS MONITORING with Autoencoder
% Detect anomalies per window from the reconstruction error
for v = 1:numel(VARIANTS)
    variantName = VARIANTS{v};
    netE1=results.(variantName).netE;  % trained encoder of this variant
    netD1=results.(variantName).netD;  % trained decoder of this variant
        %% Loop for kpi indicators calculation
        % for jj=1:size(XTest,4)
        %     dlX = dlarray(single(XTest(:,:,:,jj)), 'SSCB');
        %     Z(:,jj)=forward(netE1, dlX);       % Projection to the latent space
        %     ZZ(:,jj)=forward(netD1, Z);         % Reconstruction from the latent space
        %     XTesthat(:,:,:,jj)=gather(extractdata(ZZ(:,jj)));  % converts dlarray → array
        % 
        % end
        % Phase 1: full forward pass
        for jj = 1:NTest
            dlX = dlarray(single(XTest(:,:,:,jj)), 'SSCB');  % one test window as dlarray
            Z_temp = forward(netE, dlX);  % encode -> latent vector
            Z(:,jj) = extractdata(Z_temp);
            
            %Zg_temp = spec.map_to_manifold(dlarray(Z(:,jj), 'CB'), Rparam);
            Xhat_temp = forward(netD, Z(:,jj));  % decode -> reconstructed images
            XTesthat(:,:,:,jj) = gather(extractdata(Xhat_temp));
        end
        % Phase 2: compute reconstructed embeddings as
        % zhat=encoder(Xhat)
        for jj = 1:NTest
            dlXhat = dlarray(single(XTesthat(:,:,:,jj)), 'SSCB');
            ZTesthat(:,jj) = extractdata(forward(netE, dlXhat));
        end


        [kpis_original, kpis_geometric, infoKPIs]=kpi_error_multichannel_images_geom(XTest,XTesthat,ZTest,ZTesthat);  % KPIs in image and latent space

        %% Thresholds for anomalies detection
        gamma=98;  % percentile used as threshold
        mseThreshold2 = prctile(MSE1(:), gamma);  % MSE threshold
        maeThreshold2 = prctile(MAE1(:), gamma);  % MAE threshold
        dssimThreshold2 = prctile(dSSIM1(:), gamma);  % dSSIM threshold
        
        
        for jj=1:size(XTest,4)           
            %% calculates if jj is failure or normal
            if MSE1(jj)>=mseThreshold2 || MAE1(jj)>=maeThreshold2 || dSSIM1(jj)>=dssimThreshold2
                state_GeomAE(jj)=1;  % anomaly
            else
                state_GeomAE(jj)=0;  % normal
            end
        end
        
        [accuracy,precision,recall,f1_score]=final_result_summary(target,state_GeomAE)  % compare with the true labels

        %% Store results
        results2.(variantName) = struct();
        results2.(variantName).accuracy = accuracy;
        results2.(variantName).precision = precision;
        results2.(variantName).recall = recall;
        results2.(variantName).f1_score = f1_score;
        results2.(variantName).MSE1 = MSE1;
        results2.(variantName).MAE1 = MAE1;
        results2.(variantName).dSSIM1 = dSSIM1;
        results2.(variantName).mseThreshold2 = mseThreshold2;
        results2.(variantName).maeThreshold2 = maeThreshold2;
        results2.(variantName).dssimThreshold2 = dssimThreshold2;
        results2.(variantName).state_GeomAE = state_GeomAE;
        results2.(variantName).target = target;
        
       
        % Add to consolidated KPI table
        kpiRow2 = struct2table(struct( ...
            'variant', string(variantName), ...
            'accuracy', accuracy, ...
            'precision', precision, ...
            'recall', recall, ...
            'f1_score', f1_score, ...
            'avg_MSE', mean(MSE1), ...
            'avg_MAE', mean(MAE1), ...
            'avg_dSSIM', mean(dSSIM1), ...
            'mseThreshold2', mseThreshold2, ...
            'maeThreshold2', maeThreshold2, ...
            'dssimThreshold2', dssimThreshold2 ...
        ));


        excelFile2 = 'resultados_summary_2.xlsx';  % output Excel file
        % Sheet name = variant name
        sheetName2 = char(string(variantName));   % in case it comes as a string
        % Save this variant's KPI row in its own sheet
        % (overwrites the sheet if it exists; otherwise creates it)
        writetable(kpiRow2, excelFile2,'Sheet', sheetName2,'WriteMode', 'overwrite');

end
% %% ========================================================================
% % SECTION 5: PROCESS MONITORING (PCA/PLS/KPCA/KPLS)
% % =========================================================================
% if CONFIG.USE_PROCESS_MONITORING
%     fprintf('\n=== Ejemplo 1: KPLS con kernel RBF / Example 1: KPLS with RBF kernel ===\n\n');
%     index1=[1:4 6:8];
%     XtrainPM=Xtrain(:,index1);
%     XtestPM=Xtest(:,index1);
%     YtrainPM=Xtrain(:,5);
%     YtestPM=Xtest(:,5);
%     [results_kpls, metrics_kpls, limits_kpls] = process_monitoring(...
%         XtrainPM, YtrainPM, XtestPM, YtestPM, ref_state_test, ...
%         'Method', 'KPLS', ...
%         'NumComponents', 3, ...
%         'Kernel', 'rbf', ...
%         'Gamma', [], ... % Use default value
%         'ConfT2', 99, ...
%         'ConfSPE', 99);
% 
%     % Show results
%     fprintf('\nLímites de confianza / Confidence limits (KPLS):\n');
%     fprintf('  T2:    %.4f\n', limits_kpls.T2);
%     fprintf('  SPE_X: %.4f\n', limits_kpls.SPE_X);
%     fprintf('  SPE_Y: %.4f\n', limits_kpls.SPE_Y);
% 
%     %% 3. Example 2: KPCA with RBF kernel
%     fprintf('\n=== Ejemplo 2: KPCA con kernel RBF / Example 2: KPCA with RBF kernel ===\n\n');
% 
%     [results_kpca, metrics_kpca, limits_kpca] = process_monitoring(...
%         XtrainPM, [], XtestPM, [], ref_state_test, ...
%         'Method', 'KPCA', ...
%         'NumComponents', 3, ...
%         'Kernel', 'rbf', ...
%         'ConfT2', 99, ...
%         'ConfSPE', 99);
% 
%     % Show results
%     fprintf('\nLímites de confianza / Confidence limits (KPCA):\n');
%     fprintf('  T2:  %.4f\n', limits_kpca.T2);
%     fprintf('  SPE: %.4f\n', limits_kpca.SPE);
% 
%     %% 4. Example 3: KPLS with polynomial kernel
%     fprintf('\n=== Ejemplo 3: KPLS con kernel polinomial / Example 3: KPLS with polynomial kernel ===\n\n');
% 
%     [results_poly, metrics_poly, limits_poly] = process_monitoring(...
%         XtrainPM, YtrainPM, XtestPM, YtestPM, ref_state_test, ...
%         'Method', 'KPLS', ...
%         'NumComponents', 2, ...
%         'Kernel', 'poly', ...
%         'Degree', 3, ...
%         'Coef', 1, ...
%         'ConfT2', 95, ...
%         'ConfSPE', 95);
% 
%     %% 5. Compare metrics
%     fprintf('\n=== Comparación de Métodos / Method Comparison ===\n\n');
% 
%     comparison_table = table(...
%         [metrics_kpls.Accuracy; metrics_kpca.Accuracy; metrics_poly.Accuracy], ...
%         [metrics_kpls.Recall; metrics_kpca.Recall; metrics_poly.Recall], ...
%         [metrics_kpls.Precision; metrics_kpca.Precision; metrics_poly.Precision], ...
%         [metrics_kpls.F1_Score; metrics_kpca.F1_Score; metrics_poly.F1_Score], ...
%         'VariableNames', {'Accuracy', 'Recall', 'Precision', 'F1_Score'}, ...
%         'RowNames', {'KPLS_RBF', 'KPCA_RBF', 'KPLS_Poly'});
% 
%     disp(comparison_table);
% 
%     %% 6. Plot confusion matrix for KPLS
%     figure('Position', [100, 100, 800, 600]);
% 
%     % Convert to numeric for confusionmat
%     y_true_num = cellfun(@(x) strcmp(x, 'Normal'), labels_test);
%     y_pred_num = cellfun(@(x) strcmp(x, 'Normal'), results_kpls.Final_Status);
% 
%     cm = confusionchart(y_true_num, y_pred_num, ...
%         'RowSummary', 'row-normalized', ...
%         'ColumnSummary', 'column-normalized');
%     cm.XLabel = 'Predicción / Prediction';
%     cm.YLabel = 'Verdadero / True';
%     cm.Title = 'Matriz de Confusión / Confusion Matrix - KPLS (1=Normal, 0=Outlier)';
% 
%     %% 7. Save results
%     fprintf('\n=== Guardando resultados / Saving results ===\n\n');
% 
%     % Create struct with all results
%     all_results = struct();
%     all_results.KPLS_RBF = struct('results', results_kpls, 'metrics', metrics_kpls, 'limits', limits_kpls);
%     all_results.KPCA_RBF = struct('results', results_kpca, 'metrics', metrics_kpca, 'limits', limits_kpca);
%     all_results.KPLS_Poly = struct('results', results_poly, 'metrics', metrics_poly, 'limits', limits_poly);
% 
%     save('monitoring_results.mat', 'all_results');
% 
%     fprintf('Resultados guardados en / Results saved to monitoring_results.mat\n');
%     fprintf('\n=== Proceso completado exitosamente / Process completed successfully ===\n');
% 
%     fprintf('\nProcess monitoring complete.\n');
% end

%% ========================================================================
% SECTION 6: EXPLAINABILITY ANALYSIS (SHAP/LIME)
% =========================================================================
if CONFIG.USE_EXPLAINABILITY
    fprintf('\n========================================\n');
    fprintf('EXPLAINABILITY ANALYSIS\n');
    fprintf('========================================\n');
    
    explainResults = struct();  % results per variant
    
    for v = 1:min(3, numel(VARIANTS))  % Limit to 3 variants for demo
        variantName = VARIANTS{v};
        fprintf('\n  Analyzing %s...\n', variantName);
        
        netE = results.(variantName).netE;
        netD = results.(variantName).netD;
        spec = results.(variantName).spec;
        Rparam = results.(variantName).Rparam;
        
        % Select sample for local explanation
        sampleIdx = 1;
        X_sample = XTest(:,:,:,sampleIdx);
        
        fprintf('    - Computing SHAP local...\n');
        shapLocal = computeSHAPLocal(X_sample, netE, netD, spec, Rparam);
        
        fprintf('    - Computing SHAP global...\n');
        shapGlobal = computeSHAPGlobal(XTest(:,:,:,1:min(50,NTest)), netE, netD, spec, Rparam);
        
        fprintf('    - Computing LIME...\n');
        limeExplain = computeLIME(X_sample, netE, netD, spec, Rparam);
        
        explainResults.(variantName) = struct( ...
            'shapLocal', shapLocal, ...
            'shapGlobal', shapGlobal, ...
            'lime', limeExplain ...
        );
        
        % Save visualization
        if CONFIG.SAVE_RESULTS
            saveFigure(shapLocal.fig, sprintf('results/explainability/%s_shap_local.png', variantName));
            saveFigure(shapGlobal.fig, sprintf('results/explainability/%s_shap_global.png', variantName));
            saveFigure(limeExplain.fig, sprintf('results/explainability/%s_lime.png', variantName));
        end
    end
    
    fprintf('\nExplainability analysis complete.\n');
end

%% ========================================================================
% SECTION 7: MONTE CARLO CLASSIFICATION ANALYSIS
% =========================================================================
if CONFIG.USE_MONTE_CARLO
    fprintf('\n========================================\n');
    fprintf('MONTE CARLO CLASSIFICATION ANALYSIS\n');
    fprintf('========================================\n');
    
    mcResults = struct();  % MC results per variant
    
    for v = 1:numel(VARIANTS)
        variantName = VARIANTS{v};
        fprintf('\n  Running MC for %s (%d trials)...\n', variantName, CONFIG.MONTE_CARLO_RUNS);
        
        netE = results.(variantName).netE;
        netD = results.(variantName).netD;
        spec = results.(variantName).spec;
        Rparam = results.(variantName).Rparam;
        
        % Run Monte Carlo trials
        mcStats = runMonteCarloClassification(XTest, netE, netD, spec, Rparam, ...
                                              CONFIG.MONTE_CARLO_RUNS);
        
        mcResults.(variantName) = mcStats;
        
        % Print mean +/- std of each metric
        fprintf('    Mean Accuracy: %.4f ± %.4f\n', mcStats.accuracy_mean, mcStats.accuracy_std);
        fprintf('    Mean F1-Score: %.4f ± %.4f\n', mcStats.f1_mean, mcStats.f1_std);
        fprintf('    Mean Precision: %.4f ± %.4f\n', mcStats.precision_mean, mcStats.precision_std);
        fprintf('    Mean Recall: %.4f ± %.4f\n', mcStats.recall_mean, mcStats.recall_std);
    end
    
    fprintf('\nMonte Carlo analysis complete.\n');
end

%% ========================================================================
% SECTION 8: FINAL KPI CONSOLIDATION & REPORTING
% =========================================================================
fprintf('\n========================================\n');
fprintf('FINAL KPI CONSOLIDATION\n');
fprintf('========================================\n');

% Compute comprehensive KPIs
finalKPIs = computeFinalKPIs(results, pmResults, mcResults, allGeometricKPIs);

% Display summary table
fprintf('\nSummary of All Variants:\n');
disp(finalKPIs.summary);

% Save all results
if CONFIG.SAVE_RESULTS
    fprintf('\nSaving results...\n');
    save('results/unified_results_complete.mat', 'results', 'pmResults', ...
         'explainResults', 'mcResults', 'finalKPIs', 'CONFIG', '-v7.3');
    writetable(finalKPIs.summary, 'results/final_kpis_summary.xlsx');
    fprintf('  Saved to: results/\n');
end

% Generate final plots
fprintf('\nGenerating final comparison plots...\n');
generateFinalPlots(finalKPIs, allGeometricKPIs, results);

fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE\n');
fprintf('========================================\n');
fprintf('Total execution time: %.2f minutes\n', toc(sysMonitor.start_time)/60);
fprintf('Results saved to: ./results/\n\n');

%% ========================================================================
% END OF MAIN SCRIPT
% =========================================================================







% Visualize
% visualize_all_methods(X1, T1, dT1, ddT1, 'Señal Sinusoidal con Ruido / Noisy Sine Signal');

% 
% %% ========================================================================
% %% IMAGE STATISTICS
% %% ========================================================================
% 
% fprintf('========================================\n');
% fprintf('ESTADÍSTICAS DE IMÁGENES GENERADAS / GENERATED IMAGE STATISTICS\n');
% fprintf('========================================\n\n');
% 
% for method = 1:8
%     fprintf('Método / Method: %s\n', method_names{method});
%     fprintf('----------------------------------------\n');
% 
%     % Original
%     img_orig = T1(:,:,method);
%     fprintf('  Original:  Min=%.4f, Max=%.4f, Media/Mean=%.4f, Std=%.4f\n', ...
%            min(img_orig(:)), max(img_orig(:)), mean(img_orig(:)), std(img_orig(:)));
% 
%     % First derivative
%     img_d1 = dT1(:,:,method);
%     fprintf('  1ª Deriv. / 1st Deriv.: Min=%.4f, Max=%.4f, Media/Mean=%.4f, Std=%.4f\n', ...
%            min(img_d1(:)), max(img_d1(:)), mean(img_d1(:)), std(img_d1(:)));
% 
%     % Second derivative
%     img_d2 = ddT1(:,:,method);
%     fprintf('  2ª Deriv. / 2nd Deriv.: Min=%.4f, Max=%.4f, Media/Mean=%.4f, Std=%.4f\n', ...
%            min(img_d2(:)), max(img_d2(:)), mean(img_d2(:)), std(img_d2(:)));
% 
%     fprintf('\n');
% end
% 
% %% ========================================================================
% %% RESULTS EXPORT (OPTIONAL)
% %% ========================================================================
% 
% fprintf('========================================\n');
% fprintf('OPCIONES DE EXPORTACIÓN / EXPORT OPTIONS\n');
% fprintf('========================================\n\n');
% 
% % Ask whether to export
% fprintf('¿Desea exportar las imágenes generadas? / Export generated images? (s/n): ');
% % user_input = input('', 's');
% 
% % For automatic execution:
% user_input = 'n';
% 
% if strcmpi(user_input, 's')
%     output_folder = 'output_complete_images';
%     if ~exist(output_folder, 'dir')
%         mkdir(output_folder);
%     end
% 
%     % Save MAT file with all results
%     save(fullfile(output_folder, 'resultados_completos.mat'), ...
%          'T1', 'dT1', 'ddT1', 'T2', 'dT2', 'ddT2', 'T3', 'dT3', 'ddT3', ...
%          'X1', 'X2', 'X3', 'fs', 't');
% 
%     fprintf('✓ Resultados guardados en / Results saved to: %s/resultados_completos.mat\n', output_folder);
% 
%     % Save individual images of Example 1
%     for method = 1:8
%         % Original
%         imwrite(mat2gray(T1(:,:,method)), ...
%                fullfile(output_folder, sprintf('%s_original.png', method_names{method})));
% 
%         % First derivative
%         imwrite(mat2gray(dT1(:,:,method)), ...
%                fullfile(output_folder, sprintf('%s_1deriv.png', method_names{method})));
% 
%         % Second derivative
%         imwrite(mat2gray(ddT1(:,:,method)), ...
%                fullfile(output_folder, sprintf('%s_2deriv.png', method_names{method})));
%     end
% 
%     fprintf('✓ Imágenes guardadas en / Images saved to: %s/\n', output_folder);
% end
% 
% %% ========================================================================
% %% FINAL SUMMARY
% %% ========================================================================
% 
% fprintf('\n========================================\n');
% fprintf('RESUMEN DEL PROCESAMIENTO / PROCESSING SUMMARY\n');
% fprintf('========================================\n\n');
% 
% fprintf('Ejemplos procesados / Processed examples: 3\n');
% fprintf('Tiempo total / Total time: %.3f s\n', tiempo1 + tiempo2 + tiempo3);
% fprintf('Tiempo promedio por ejemplo / Average time per example: %.3f s\n', (tiempo1 + tiempo2 + tiempo3)/3);
% fprintf('\nImágenes generadas por ejemplo / Images per example: 24 (8 métodos/methods × 3 señales/signals)\n');
% fprintf('Total de imágenes generadas / Total images generated: 72\n\n');
% 
% fprintf('Métodos implementados / Implemented methods:\n');
% for i = 1:length(method_names)
%     fprintf('  %d. %s\n', i, method_names{i});
% end
% 
% fprintf('\n========================================\n');
% fprintf('✓ Procesamiento completo finalizado / Processing completed\n');
% fprintf('========================================\n');

%% ========================================================================
%% VISUALIZATION HELPER FUNCTION
%% ========================================================================

function visualize_all_methods(X, T, dT, ddT, titulo)
    % Plots all images generated for a signal
    
    method_names = {'GASF', 'GADF', 'MTF', 'RP', 'SPEC', 'SCALO', 'GRAY', 'SSM'};
    
    % Compute derivatives for plotting
    N = length(X);  % signal length
    dX = [X(1); diff(X)];  % 1st derivative (finite difference)
    ddX = [X(1); dX(2); diff(dX(2:end))];  % 2nd derivative
    
    figure('Name', titulo, 'Position', [100, 100, 1600, 900], 'Color', 'w');
    
    % Top panel: original signals
    subplot(4, 8, [1 2 3]);
    plot(X, 'b-', 'LineWidth', 1.5);
    grid on;
    title('Señal Original / Original Signal');
    ylabel('Amplitud / Amplitude');
    
    subplot(4, 8, [4 5 6]);
    plot(dX, 'r-', 'LineWidth', 1.5);
    grid on;
    title('Primera Derivada / First Derivative');
    
    subplot(4, 8, [7 8]);
    plot(ddX, 'Color', [0.85, 0.33, 0.10], 'LineWidth', 1.5);
    grid on;
    title('Segunda Derivada / Second Derivative');
    
    % Images: 3 rows (original, 1st deriv, 2nd deriv) × 8 methods
    for method = 1:8
        % Row 2: original signal
        subplot(4, 8, 8 + method);
        imagesc(T(:,:,method));
        axis square off;
        if method == 1
            ylabel('Original', 'FontWeight', 'bold');
        end
        title(method_names{method});
        colorbar;
        
        % Row 3: first derivative
        subplot(4, 8, 16 + method);
        imagesc(dT(:,:,method));
        axis square off;
        if method == 1
            ylabel('1ª Deriv. / 1st Deriv.', 'FontWeight', 'bold');
        end
        colorbar;
        
        % Row 4: second derivative
        subplot(4, 8, 24 + method);
        imagesc(ddT(:,:,method));
        axis square off;
        if method == 1
            ylabel('2ª Deriv. / 2nd Deriv.', 'FontWeight', 'bold');
        end
        colorbar;
    end
    
    sgtitle(titulo, 'FontSize', 14, 'FontWeight', 'bold');
end


%% ========================================================================
% DATA LOADING
% =========================================================================
function [XTrain, XTest, YTrain, YTest, imageSize] = load_dataset()
    % Load or generate dataset
    if exist('multich_images_boiler_sept2025.mat', 'file')
        load('multich_images_boiler_sept2025.mat');
        fprintf('  Loaded: multich_images_boiler_sept2025.mat\n');
    elseif exist('multich_images_boiler.mat', 'file')
        data = load('multich_images_boiler.mat');
        XTrain = data.XTrain;
        XTest = data.XTest;
        fprintf('  Loaded: multich_images_boiler.mat\n');
    else
        fprintf('  Generating synthetic dataset...\n');
        [XTrain, XTest] = generateSyntheticDataset();
    end
    
    % Ensure single precision
    XTrain = single(XTrain);
    XTest = single(XTest);
    
    [H, W, C, NTrain] = size(XTrain);
    NTest = size(XTest, 4);
    imageSize = [H, W, C];
    
    % Generate physics labels if not present
    if exist('eta_dir', 'var') && exist('eta_ind', 'var')
        YTrain = [eta_dir(:), eta_ind(:)];
        YTest = 0.75 + 0.05*randn(NTest, 2, 'single');
    else
        YTrain = 0.75 + 0.05*randn(NTrain, 2, 'single');
        YTest = 0.75 + 0.05*randn(NTest, 2, 'single');
    end
end

function [XTrain, XTest] = generateSyntheticDataset()
    % Generate synthetic multispectral images
    H = 32; W = 32; C = 6;  % image size
    NTrain = 500; NTest = 100;  % number of samples
    
    % Preallocate
    XTrain = zeros(H, W, C, NTrain, 'single');
    XTest = zeros(H, W, C, NTest, 'single');
    
    for i = 1:NTrain
        XTrain(:,:,:,i) = generateSample(H, W, C);
    end
    for i = 1:NTest
        XTest(:,:,:,i) = generateSample(H, W, C);
    end
    
    % Nested helper: smooth gradients + random Gaussian blob + noise
    function X = generateSample(H, W, C)
        [xx, yy] = meshgrid(linspace(-1,1,W), linspace(-1,1,H));
        X = zeros(H, W, C, 'single');
        for k = 1:C
            ang = 2*pi*(k/C);  % different gradient direction per channel
            X(:,:,k) = 0.5*xx*cos(ang) + 0.5*yy*sin(ang);
        end
        % Add blob
        cx = 0.5*randn; cy = 0.5*randn; sig = 0.2 + 0.1*rand;
        blob = exp(-((xx-cx).^2 + (yy-cy).^2)/(2*sig^2));
        k0 = randi(C);
        X(:,:,k0) = X(:,:,k0) + 0.7*single(blob);
        X = X + 0.05*randn(size(X), 'single');  % add noise
        X = max(min(X, 1), -1);  % clip to [-1, 1]
    end
end

%% ========================================================================
% NETWORK BUILDING
% =========================================================================
function [netE, netD, encShape] = buildAutoencoder(imageSize, latentDim, hiddenSize, spec)
H = imageSize(1); W = imageSize(2); C = imageSize(3);  % input image dimensions

% Encoder: 2 conv blocks (conv-BN-LeakyReLU-pool) + FC to latent space
layersE = [
    imageInputLayer(imageSize, Normalization='none')
    convolution2dLayer(3, 2*hiddenSize, Padding='same', WeightsInitializer='narrow-normal')  % conv block 1
    batchNormalizationLayer
    leakyReluLayer(0.05)
    maxPooling2dLayer(2,Stride=2)  % downsample x2
    convolution2dLayer(3, hiddenSize, Padding='same', WeightsInitializer='narrow-normal')  % conv block 2
    batchNormalizationLayer
    leakyReluLayer(0.05)
    maxPooling2dLayer(2,Stride=2)  % downsample x2
    fullyConnectedLayer(latentDim)       % Euclidean pre-latent
    functionLayer(@(Z) Z, Name='IdentityPreManifold')  % keep explicit
];
netE = dlnetwork(layerGraph(layersE));  % build dlnetwork

encH = H/2/2; encW = W/2/2; encC = hiddenSize;  % spatial size after two poolings
decoderInputSize = encH*encW*encC;  % flattened size

% Decoder: FC -> reshape -> transposed convs back to image size
layersD = [
    featureInputLayer(latentDim)
    functionLayer(@(Y) Y, Name='IdentityFromManifold') % in case we wanted inverse maps
    fullyConnectedLayer(decoderInputSize)  % expand latent vector
    leakyReluLayer(0.05)
    reshapeLayer2([encH, encW, encC], 'reshape')  % vector -> [encH x encW x encC]
    transposedConv2dLayer(3, hiddenSize, Stride=2, Cropping='same', WeightsInitializer='narrow-normal')  % upsample x2
    leakyReluLayer(0.05)
    transposedConv2dLayer(3, 2*hiddenSize, Stride=2, Cropping='same', WeightsInitializer='narrow-normal')  % upsample x2
    leakyReluLayer(0.05)
    transposedConv2dLayer(3, C, Cropping='same', WeightsInitializer='narrow-normal')  % output with C channels
    tanhLayer  % output in [-1, 1]
];
netD = dlnetwork(layerGraph(layersD));  % build dlnetwork

encShape = [encH, encW, encC];
end

function Y = reshapeForDecoder(X, encShape)
    % Reshape flat vector to spatial tensor
    Y = reshape(X, [encShape, size(X, 4)]);
end

function physInDim = getPhysicsInputDim(spec)
    % Determine input dimension for physics heads
    physInDim = spec.latentDim;
end

function [netDir, netInd] = buildPhysicsHeads(inputDim)
    % Build physics prediction heads (for PINN)
    % Direct efficiency head
    layersDir = [
        featureInputLayer(inputDim, 'Name', 'z_in')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(32, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1, 'Name', 'eta_dir')
    ];
    netDir = dlnetwork(layerGraph(layersDir));
    
    % Indirect efficiency head
    layersInd = [
        featureInputLayer(inputDim, 'Name', 'z_in')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(32, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1, 'Name', 'eta_ind')
    ];
    netInd = dlnetwork(layerGraph(layersInd));
end

%% ========================================================================
% VARIANT SPECIFICATIONS
% =========================================================================
function spec = getVariantSpec(name)
    % Return specification for each variant
    % Default parameters shared by all variants
    baseSpec = struct( ...
        'latentDim', 16, ...
        'hiddenSize', 32, ...
        'learnRate', 1.0, ...
        'R', 1.0, ...
        'geo_cap', 10.0, ...
        'weight_override', struct() ...
    );
    
    switch name
        case 'EUCLIDEAN'  % flat latent space (no projection)
            spec = baseSpec;
            spec.map_to_manifold = @(Z, R) Z;
            spec.geo_distance = @(Z1, Z2, R) sqrt(sum((Z1-Z2).^2, 1));
            spec.clampR = @(R) R;
            
        case 'POINCARE'  % hyperbolic: Poincare ball model
            spec = baseSpec;
            spec.map_to_manifold = @proj_poincare;
            spec.geo_distance = @dist_poincare;
            spec.clampR = @(R) max(R, 0.1);
            
        case 'LORENTZ'  % hyperbolic: hyperboloid (Lorentz) model
            spec = baseSpec;
            spec.latentDim = 17; % n+1 for hyperboloid
            spec.map_to_manifold = @proj_lorentz;
            spec.geo_distance = @dist_lorentz;
            spec.clampR = @(R) max(R, 0.1);
            
        case 'KLEIN'  % hyperbolic: Klein model (distance via Poincare)
            spec = baseSpec;
            spec.map_to_manifold = @proj_klein;
            spec.geo_distance = @dist_klein_via_poincare;
            spec.clampR = @(R) max(R, 0.1);
            
        case 'SPHERE'  % sphere of radius R=1
            spec = baseSpec;
            spec.R = 1.0;
            spec.map_to_manifold = @proj_sphere;
            spec.geo_distance = @dist_sphere;
            spec.clampR = @(R) max(min(R, 10), 0.1);
            
        case 'SPHERE_R'  % sphere with initial radius R=2
            spec = baseSpec;
            spec.R = 2.0;
            spec.map_to_manifold = @proj_sphere;
            spec.geo_distance = @dist_sphere;
            spec.clampR = @(R) max(min(R, 10), 0.1);
            
        case 'PRODUCT_EHS'  % product space: Euclidean x Hyperbolic x Sphere
            spec = baseSpec;
            spec.latentDim = 18; % Divisible by 3
            spec.map_to_manifold = @proj_mixed_EHS;
            spec.geo_distance = @dist_mixed_EHS;
            spec.clampR = @(R) max(R, 0.1);
            
        case 'H2_PRODUCT'  % product of two hyperbolic spaces
            spec = baseSpec;
            spec.latentDim = 16;
            spec.map_to_manifold = @proj_H2xH2;
            spec.geo_distance = @dist_H2xH2;
            spec.clampR = @(R) max(R, 0.1);
            
        case 'ELLIPTIC'  % elliptic space (sphere with antipodes identified)
            spec = baseSpec;
            spec.map_to_manifold = @proj_sphere;
            spec.geo_distance = @dist_elliptic;
            spec.clampR = @(R) max(min(R, 10), 0.1);
            
        case 'GEODESIC_TANGENT'  % exponential map at the Poincare origin
            spec = baseSpec;
            spec.map_to_manifold = @expmap_poincare_origin;
            spec.geo_distance = @dist_poincare;
            spec.clampR = @(R) max(R, 0.1);
            
        case 'VAE'  % variational autoencoder
            spec = baseSpec;
            spec.latentDim = 32; % mu + logvar
            spec.map_to_manifold = @vae_reparameterize;
            spec.geo_distance = @(Z1, Z2, R) sqrt(sum((Z1-Z2).^2, 1));
            spec.clampR = @(R) R;
            spec.weight_override.kl = 1e-3;
            
        case 'CONV_AE'  % plain conv autoencoder (no geometric loss)
            spec = baseSpec;
            spec.map_to_manifold = @(Z, R) Z;
            spec.geo_distance = @(Z1, Z2, R) sqrt(sum((Z1-Z2).^2, 1));
            spec.clampR = @(R) R;
            spec.weight_override.geo = 0.0;
            
        case 'ROBUST_AE'  % robust AE (lower reconstruction weight)
            spec = baseSpec;
            spec.map_to_manifold = @(Z, R) Z;
            spec.geo_distance = @(Z1, Z2, R) sqrt(sum((Z1-Z2).^2, 1));
            spec.clampR = @(R) R;
            spec.weight_override.rec = 0.5;
            
        otherwise
            error('Unknown variant: %s', name);
    end
end

%% ========================================================================
% GEOMETRIC PROJECTIONS AND DISTANCES
% =========================================================================
% Poincare ball: rescale points to stay inside the ball of radius R
function Zp = proj_poincare(Z, R)
    eps = 1e-9;  % numerical stability
    n = sqrt(sum(Z.^2, 1)) + eps;  % norm of each column
    scale = (R - 1e-5) ./ max(n, R);  % shrink factor for points outside the ball
    Zp = Z .* min(scale, 1);
end

% Poincare geodesic distance
function D = dist_poincare(Z1, Z2, R)
    eps = 1e-9;
    diff = Z1 - Z2;
    n1 = sum(Z1.^2, 1);
    n2 = sum(Z2.^2, 1);
    ndiff = sum(diff.^2, 1);
    denom = (1 - n1./(R^2+eps)) .* (1 - n2./(R^2+eps));
    D = R .* acosh(1 + 2*ndiff./max(denom, eps));  % hyperbolic distance (acosh formula)
end

function Zl = proj_lorentz(Z, R)
    % Project to hyperboloid: x0^2 - sum(xi^2) = R^2
    d = size(Z, 1);
    Zeuc = Z(1:d-1, :);  % spatial coordinates (last row dropped)
    n2 = sum(Zeuc.^2, 1);
    x0 = sqrt(R^2 + n2);  % time coordinate so the point lies on the hyperboloid
    Zl = [x0; Zeuc];
end

% Lorentz distance via the Minkowski inner product
function D = dist_lorentz(Z1, Z2, R)
    eps = 1e-9;
    inner = Z1(1,:).*Z2(1,:) - sum(Z1(2:end,:).*Z2(2:end,:), 1);  % Minkowski inner product
    D = R .* acosh(max(inner ./ (R^2+eps), 1+eps));
end

% Klein projection reuses the Poincare projection
function Zk = proj_klein(Z, R)
    Zk = proj_poincare(Z, R);
end

% Klein distance: convert to Poincare and measure there
function D = dist_klein_via_poincare(K1, K2, R)
    P1 = klein2poincare(K1, R);
    P2 = klein2poincare(K2, R);
    D = dist_poincare(P1, P2, R);
end

% Map Klein coordinates to the Poincare ball
function P = klein2poincare(K, R)
    eps = 1e-9;
    k = K ./ (R + eps);
    nk = sum(k.^2, 1);
    Punit = k ./ (1 + sqrt(max(1 - nk, eps)));
    P = R .* Punit;
end

% Project onto the sphere of radius R
function Zs = proj_sphere(Z, R)
    eps = 1e-9;
    n = sqrt(sum(Z.^2, 1)) + eps;
    Zs = (R .* Z) ./ n;
end

% Spherical (great-circle) distance
function D = dist_sphere(U, V, R)
    eps = 1e-9;
    dotUV = sum(U.*V, 1);
    cosang = dotUV ./ max(R^2, eps);  % cosine of the angle
    cosang = min(max(cosang, -1+1e-6), 1-1e-6);  % clip to avoid acos NaN
    D = R .* acos(cosang);
end

% Elliptic distance: shortest angle considering antipodal points
function D = dist_elliptic(U, V, R)
    theta = dist_sphere(U, V, R) ./ max(R, 1e-9);
    theta = min(theta, pi - theta);
    D = R .* theta;
end

% Mixed space: split latent into Euclidean / hyperbolic / spherical thirds
function Zmix = proj_mixed_EHS(Z, R)
    d = size(Z, 1);
    d1 = floor(d/3);
    d2 = floor(2*d/3);
    Ze = Z(1:d1, :);
    Zh = proj_poincare(Z(d1+1:d2, :), R);
    Zs = proj_sphere(Z(d2+1:end, :), R);
    Zmix = [Ze; Zh; Zs];
end

% Mixed distance: sum of the distances in each subspace
function D = dist_mixed_EHS(Z1, Z2, R)
    d = size(Z1, 1);
    d1 = floor(d/3);
    d2 = floor(2*d/3);
    E1 = Z1(1:d1, :); E2 = Z2(1:d1, :);
    H1 = Z1(d1+1:d2, :); H2 = Z2(d1+1:d2, :);
    S1 = Z1(d2+1:end, :); S2 = Z2(d2+1:end, :);
    
    De = sqrt(sum((E1-E2).^2, 1));
    Dh = dist_poincare(H1, H2, R);
    Ds = dist_sphere(S1, S2, R);
    D = De + Dh + Ds;
end

% H2 x H2: two Poincare balls
function Zhh = proj_H2xH2(Z, R)
    d = size(Z, 1);
    half = floor(d/2);
    A = proj_poincare(Z(1:half, :), R);
    B = proj_poincare(Z(half+1:end, :), R);
    Zhh = [A; B];
end

% H2 x H2 distance: sum of both Poincare distances
function D = dist_H2xH2(Z1, Z2, R)
    d = size(Z1, 1);
    half = floor(d/2);
    A1 = Z1(1:half, :); B1 = Z1(half+1:end, :);
    A2 = Z2(1:half, :); B2 = Z2(half+1:end, :);
    D = dist_poincare(A1, A2, R) + dist_poincare(B1, B2, R);
end

% Exponential map at the origin: tangent vector -> Poincare ball
function Zexp = expmap_poincare_origin(V, R)
    eps = 1e-9;
    nv = sqrt(sum(V.^2, 1)) + eps;
    alpha = tanh((nv) ./ (2*R));
    Zexp = (R .* (V./nv)) .* alpha;
end

function Z = vae_reparameterize(Z_mulogvar, ~)
    % VAE reparameterization trick
    d = size(Z_mulogvar, 1);
    half = floor(d/2);
    mu = Z_mulogvar(1:half, :);
    logvar = Z_mulogvar(half+1:end, :);
    std = exp(0.5 * logvar);  % standard deviation
    eps = randn(size(mu), 'like', mu);  % random sample N(0,1)
    Z = mu + eps .* std;  % z = mu + sigma*eps
end

% Apply the variant's clamp to R
function R = clampR(R, spec)
    R = spec.clampR(R);
end

%% ========================================================================
% LOSS COMPUTATION
% =========================================================================
function [L, grads, parts] = computeLoss(X, ydir, yind, netE, netD, ...
                                         netDir, netInd, R, beta, spec, prev, usePhysics)
    % Unified loss computation for all variants
    
    % Forward pass through encoder
    Z = forward(netE, X);
    
    % Project to manifold
    Zg = spec.map_to_manifold(Z, R);
    
    % Decoder reconstruction
    Xhat = forward(netD, Zg);
    
    % 1) Reconstruction loss (MSE + Cauchy for robustness)
    Lmse = mse(Xhat, X);
    Lcauchy = cauchyLoss(Xhat, X, 5.0);
    Lrec = 0.5*Lmse + 0.5*Lcauchy;
    
    % 2) Geometric consistency loss
    Ztrue = spec.map_to_manifold(forward(netE, X), R);
    Zrec = spec.map_to_manifold(forward(netE, Xhat), R);
    Dgeo = spec.geo_distance(Ztrue, Zrec, R);  % distance between original and re-encoded latent
    Lgeo = mean(tanh(min(Dgeo, spec.geo_cap)), 'all');  % bounded geometric loss
    
    % 3) Latent norm penalty
    Lnorm = mean(sum(Zg.^2, 1), 'all');
    
    % 4) L2 regularization
    Ll2 = sum(cellfun(@(v) sum(v.^2, 'all'), netE.Learnables.Value)) + ...
          sum(cellfun(@(v) sum(v.^2, 'all'), netD.Learnables.Value));
    
    % 5) Physics loss (PINN)
    if usePhysics && ~isempty(netDir) && ~isempty(ydir)
        eta_dir_pred = forward(netDir, Zg);
        eta_ind_pred = forward(netInd, Zg);
        Lphys = mse(eta_dir_pred, ydir) + mse(eta_ind_pred, yind);
    else
        Lphys = dlarray(0);
    end
    
    % 6) Temporal variability loss
    if ~isempty(prev.E)
        try
            Zprev = spec.map_to_manifold(forward(prev.E, X), R);
            Lvar = mean(sum((Zg - Zprev).^2, 1), 'all');
        catch
            Lvar = dlarray(0);
        end
    else
        Lvar = dlarray(0);
    end
    
    % 7) KL divergence (for VAE)
    if contains(func2str(spec.map_to_manifold), 'vae')
        d = size(Z, 1);
        half = floor(d/2);
        mu = Z(1:half, :);
        logvar = Z(half+1:end, :);
        Lkl = -0.5 * mean(1 + logvar - mu.^2 - exp(logvar), 'all');  % KL divergence vs N(0,I)
    else
        Lkl = dlarray(0);
    end
    
    % Total loss
    L = beta(1)*Lrec + beta(2)*Ll2 + beta(3)*Lnorm + ...
        beta(4)*Lgeo + beta(5)*Lphys + beta(6)*Lvar + beta(7)*Lkl;
    
    % Compute gradients
    grads = struct();
    grads.gE = dlgradient(L, netE.Learnables, 'RetainData', true);
    grads.gD = dlgradient(L, netD.Learnables, 'RetainData', true);
    grads.gR = dlgradient(L, R, 'RetainData', true);
    
    if usePhysics && ~isempty(netDir)
        grads.gDir = dlgradient(L, netDir.Learnables, 'RetainData', true);
        grads.gInd = dlgradient(L, netInd.Learnables, 'RetainData', true);
    else
        grads.gDir = [];
        grads.gInd = [];
    end
    
    % Gradient clipping
    grads.gE = clipGradients(grads.gE, 5.0);
    grads.gD = clipGradients(grads.gD, 5.0);
    grads.gR = min(max(grads.gR, -2.0), 2.0);
    
    % Return loss components
    parts = struct('Lrec', Lrec, 'Lgeo', Lgeo, 'Lphys', Lphys, ...
                   'Lvar', Lvar, 'Lkl', Lkl);
end

% Cauchy loss: robust to outliers (c = scale)
function L = cauchyLoss(Xhat, X, c)
    E = Xhat - X;
    L = mean(log(1 + (E./c).^2), 'all');
end

% Rescale each gradient whose L2 norm exceeds the threshold
function G = clipGradients(G, threshold)
    for i = 1:size(G, 1)
        g = G.Value{i};
        n = sqrt(sum(g(:).^2));
        if n > threshold
            G.Value{i} = g * (threshold / n);
        end
    end
end

%% ========================================================================
% MINIBATCH QUEUE PREPARATION
% =========================================================================
function mbq = prepareMinibatchQueue(X, ydir, yind, batchSize, usePhysics)
    dsX = arrayDatastore(X, 'IterationDimension', 4);  % datastore over images (4th dim = samples)
    
    % With physics: combine images + 2 targets
    if usePhysics && ~isempty(ydir)
        ydir4 = reshape(single(ydir), [1 1 1 numel(ydir)]);
        yind4 = reshape(single(yind), [1 1 1 numel(yind)]);
        dsDir = arrayDatastore(ydir4, 'IterationDimension', 4);
        dsInd = arrayDatastore(yind4, 'IterationDimension', 4);
        ds = combine(dsX, dsDir, dsInd);
        mbq = minibatchqueue(ds, 3, 'MiniBatchSize', batchSize, ...
            'MiniBatchFcn', @(X,Y,Z) {dlarray(single(X{1}),'SSCB'), ...
                                      dlarray(single(Y{1}),'CB'), ...
                                      dlarray(single(Z{1}),'CB')}, ...
            'MiniBatchFormat', {'SSCB','CB','CB'});
    else
        % Without physics: images only
        mbq = minibatchqueue(dsX, 1, 'MiniBatchSize', batchSize, ...
            'MiniBatchFcn', @(X) dlarray(single(X{1}),'SSCB'), ...
            'MiniBatchFormat', 'SSCB');
    end
end

%% ========================================================================
% EVALUATION
% =========================================================================
function [testResults, geoKPIs] = evaluateModel(XTest, ydir_test, yind_test, ...
    netE, netD, netDir, netInd, Rparam, spec, usePhysics)
    
    NTest = size(XTest, 4);
    dlX = dlarray(single(XTest), 'SSCB');  % whole test set as one batch
    
    % Forward pass
    Z = forward(netE, dlX);
    Zg = spec.map_to_manifold(Z, Rparam);
    Xhat = forward(netD, Zg);
    
    % Compute metrics
    Z_cpu = gather(Zg);
    Xhat_cpu = gather(Xhat);
    X_cpu = gather(XTest);
    
    % Basic reconstruction metrics
    MSE_vals = zeros(NTest, 1);
    MAE_vals = zeros(NTest, 1);
    SSIM_vals = zeros(NTest, 1);
    
    for i = 1:NTest
        err = X_cpu(:,:,:,i) - Xhat_cpu(:,:,:,i);
        MSE_vals(i) = mean(err(:).^2);
        MAE_vals(i) = mean(abs(err(:)));
        try
            SSIM_vals(i) = mean(ssim(Xhat_cpu(:,:,:,i), X_cpu(:,:,:,i)));
        catch  % fallback if ssim is unavailable
            SSIM_vals(i) = 1 - MSE_vals(i);
        end
    end
    
    testResults = struct();
    testResults.Z = Z_cpu;
    testResults.Xhat = Xhat_cpu;
    testResults.MSE = mean(MSE_vals);
    testResults.MAE = mean(MAE_vals);
    testResults.SSIM = mean(SSIM_vals);
    
    % Geometric distances
    Zrec = spec.map_to_manifold(forward(netE, Xhat), Rparam);
    D = gather(extractdata(spec.geo_distance(Zg, Zrec, Rparam)));  % geodesic distance per sample
    
    geoKPIs = struct();
    geoKPIs.geo_dist_mean = mean(D);
    geoKPIs.geo_dist_std = std(D);
    geoKPIs.geo_dist_max = max(D);
    
    % Physics prediction metrics
    if usePhysics && ~isempty(netDir)
        eta_dir_pred = gather(extractdata(forward(netDir, Zg)));
        eta_ind_pred = gather(extractdata(forward(netInd, Zg)));
        
        geoKPIs.phys_dir_mse = mean((eta_dir_pred(:) - ydir_test(:)).^2);
        geoKPIs.phys_ind_mse = mean((eta_ind_pred(:) - yind_test(:)).^2);
    else
        geoKPIs.phys_dir_mse = NaN;
        geoKPIs.phys_ind_mse = NaN;
    end
    
    testResults.Z_train = Z_cpu; % For PM analysis
end

%% ========================================================================
% GEOMETRIC PERFORMANCE ANALYSIS
% =========================================================================
function geoAnalysis = analyzeGeometricPerformance(Z, Xhat, X, spec, R)
    % Analyze geometric properties of embeddings
    
    % 1) Curvature estimation (local)
    curvature = estimateLocalCurvature(Z);
    
    % 2) Geodesic path deviation
    deviation = computeGeodesicDeviation(Z, spec, R);
    
    % 3) Volume preservation (Jacobian determinant)
    volPres = estimateVolumePreservation(Z);
    
    geoAnalysis = struct();
    geoAnalysis.mean_curvature = mean(curvature);
    geoAnalysis.std_curvature = std(curvature);
    geoAnalysis.mean_deviation = mean(deviation);
    geoAnalysis.volume_preservation = volPres;
end

% Local curvature: deviation of k-nearest neighbors from a 2D tangent plane
function curv = estimateLocalCurvature(Z)
    if isa(Z, 'dlarray')
        Z = extractdata(Z);
    end
    Z = double(Z);  %% Ensure precision
    % Estimate curvature via local neighborhood analysis
    N = size(Z, 2);
    curv = zeros(N, 1);
    k = min(10, N-1);  % number of neighbors
    
    for i = 1:min(100, N) % Sample subset for efficiency
        zi = Z(:, i);
        dists = sqrt(sum((Z - zi).^2, 1));  % distances to all points
        [~, idx] = sort(dists);
        neighbors = Z(:, idx(2:k+1));  % k nearest neighbors (excluding itself)
        
        % Fit local tangent plane and measure deviation
        [~, ~, V] = svd(neighbors - mean(neighbors, 2), 'econ');
        proj = V(:, 1:min(2, size(V, 2)));
        %devs = sqrt(sum((neighbors - proj*(proj'*neighbors)).^2, 1));
        % Center explicitly
        neighbors_centered = neighbors - mean(neighbors, 2);
        [~, ~, V] = svd(neighbors_centered, 'econ');
        
        % Project consistently
        projected = proj * (proj' * neighbors_centered');
        devs = sqrt(sum((neighbors_centered - projected').^2, 1));
        curv(i) = mean(devs) / mean(dists(idx(2:k+1)));  % normalized deviation
    end
end

% Relative gap between geodesic and Euclidean distance of consecutive points
function dev = computeGeodesicDeviation(Z, spec, R)
    if isa(Z, 'dlarray')
        Z = extractdata(Z);
    end
    Z = double(Z);  %% Ensure precision
    % Measure deviation from geodesics
    N = size(Z, 2);
    dev = zeros(N-1, 1);
    
    for i = 1:min(50, N-1)
        z1 = Z(:, i);
        z2 = Z(:, i+1);
        
        % Direct geodesic distance
        d_geo = spec.geo_distance(z1, z2, R);
        
        % Euclidean approximation
        d_euc = sqrt(sum((z1 - z2).^2));
        
        dev(i) = abs(double(d_geo) - d_euc) / (double(d_geo) + 1e-9);
    end
end

% Volume preservation: geometric mean of latent variances
function volPres = estimateVolumePreservation(Z)
    if isa(Z, 'dlarray')
        Z = extractdata(Z);
    end
    Z = double(Z);  %%// Ensure precision
    % Estimate volume preservation (ratio of variances)
    varZ = var(Z, 0, 2);
    volPres = exp(mean(log(varZ + 1e-9)));
end

% True for variants with a curved latent space
function tf = isGeometricVariant(name)
    nonGeoVariants = {'EUCLIDEAN', 'VAE', 'CONV_AE', 'ROBUST_AE'};  % flat-space variants
    tf = ~ismember(name, nonGeoVariants);
end

%% ========================================================================
% NETWORK SNAPSHOT
% =========================================================================
% Copy of current networks (used by the temporal variability loss)
function prev = snapshotNetworks(netE, netD, netDir, netInd)
    prev = struct();
    prev.E = netE;
    prev.D = netD;
    prev.Dir = netDir;
    prev.Ind = netInd;
end

%% ========================================================================
% RESOURCE MONITORING (CROSS-PLATFORM)
% =========================================================================
% Initialize resource/energy monitoring state
function sysMonitor = initSystemMonitoring(powerModel)
    sysMonitor = struct();
    sysMonitor.start_time = tic;  % global timer
    sysMonitor.prev_t = tic;  % timer between samples
    sysMonitor.cum_energy_j = 0;  % cumulative energy (J)
    sysMonitor.powerModel = powerModel;
    sysMonitor.rapl_ok = false;  % true if Intel RAPL is available
    
    % Check RAPL availability (Linux)
    if isunix
        raplPath = '/sys/class/powercap/intel-rapl:0/energy_uj';
        if exist(raplPath, 'file') == 2
            try
                uj = str2double(fileread(raplPath));
                if ~isnan(uj)
                    sysMonitor.prev_rapl_uj = uj;
                    sysMonitor.rapl_ok = true;
                end
            catch
                % RAPL not available
            end
        end
    end
end

function [row, sysMonitor] = sampleSystemMetrics(sysMonitor, powerModel)
    % Sample CPU, GPU, RAM, Power metrics (cross-platform)
    
    % All metrics default to NaN (unavailable)
    row = struct('cpu_pct', NaN, 'ram_used_mb', NaN, 'ram_total_mb', NaN, ...
                 'ram_pct', NaN, 'gpu_util_pct', NaN, 'gpu_mem_used_mb', NaN, ...
                 'gpu_mem_total_mb', NaN, 'gpu_power_w', NaN, 'cpu_power_w', NaN, ...
                 'total_power_w', NaN, 'cum_energy_j', NaN);
    
    dt = toc(sysMonitor.prev_t);  % time since last sample
    sysMonitor.prev_t = tic;
    
    % === CPU Usage ===
    if ispc
        % Windows: use WMIC
        try
            [~, out] = system('wmic cpu get loadpercentage /value');
            tok = regexp(out, 'LoadPercentage=(\d+)', 'tokens');
            if ~isempty(tok)
                row.cpu_pct = str2double(tok{1}{1});
            end
        catch
        end
    else
        % Linux/Mac: use top
        try
            [~, out] = system('top -bn1 | grep "Cpu(s)"');
            tok = regexp(out, '(\d+\.\d+)\s+id', 'tokens');
            if ~isempty(tok)
                idle = str2double(tok{1}{1});
                row.cpu_pct = 100 - idle;
            end
        catch
        end
    end
    
    % === RAM Usage ===
    if ispc
        try
            [~, out] = system('wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value');
            t1 = regexp(out, 'TotalVisibleMemorySize=(\d+)', 'tokens');
            f1 = regexp(out, 'FreePhysicalMemory=(\d+)', 'tokens');
            if ~isempty(t1) && ~isempty(f1)
                tot_kb = str2double(t1{1}{1});
                free_kb = str2double(f1{1}{1});
                row.ram_total_mb = tot_kb / 1024;
                row.ram_used_mb = (tot_kb - free_kb) / 1024;
                row.ram_pct = 100 * row.ram_used_mb / row.ram_total_mb;
            end
        catch
        end
    else
        try
            [~, out] = system('free -m | awk ''/Mem:/ {print $2" "$3}''');
            vals = sscanf(out, '%f %f');
            if numel(vals) == 2
                row.ram_total_mb = vals(1);
                row.ram_used_mb = vals(2);
                row.ram_pct = 100 * row.ram_used_mb / row.ram_total_mb;
            end
        catch
        end
    end
    
    % === GPU Metrics (nvidia-smi) ===
    try
        [st, out] = system('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits');
        if st == 0
            % Parse CSV: utilization, mem used, mem total, power
            vals = textscan(out, '%f,%f,%f,%f');
            if ~isempty(vals) && ~isempty(vals{1})
                row.gpu_util_pct = vals{1}(1);
                row.gpu_mem_used_mb = vals{2}(1);
                row.gpu_mem_total_mb = vals{3}(1);
                row.gpu_power_w = vals{4}(1);
            end
        end
    catch
    end
    
    % Estimate GPU power if not available
    if isnan(row.gpu_power_w) && ~isnan(row.gpu_util_pct)
        row.gpu_power_w = (row.gpu_util_pct / 100) * powerModel.gpu_tdp_w;  % estimate: utilization x TDP
    end
    
    % === CPU Power (RAPL or estimation) ===
    cpu_power_w = NaN;
    if sysMonitor.rapl_ok
        raplPath = '/sys/class/powercap/intel-rapl:0/energy_uj';
        try
            uj = str2double(fileread(raplPath));
            if ~isnan(uj) && dt > 0
                duj = uj - sysMonitor.prev_rapl_uj;
                if duj < 0
                    duj = duj + 2^32; % Wrap-around
                end
                cpu_power_w = (duj * 1e-6) / dt;  % energy delta / time = power
                sysMonitor.prev_rapl_uj = uj;
                sysMonitor.cum_energy_j = sysMonitor.cum_energy_j + duj * 1e-6;  % accumulate measured energy
            end
        catch
        end
    end
    
    % Estimate CPU power if not measured
    if isnan(cpu_power_w) && ~isnan(row.cpu_pct)
        cpu_power_w = (row.cpu_pct / 100) * powerModel.cpu_tdp_w;
    end
    
    row.cpu_power_w = cpu_power_w;
    
    % === Total Power ===
    base_w = powerModel.idle_w + powerModel.other_w;  % idle + other components
    row.total_power_w = base_w;
    if ~isnan(row.cpu_power_w)
        row.total_power_w = row.total_power_w + row.cpu_power_w;
    end
    if ~isnan(row.gpu_power_w)
        row.total_power_w = row.total_power_w + row.gpu_power_w;
    end
    
    % Update cumulative energy
    if dt > 0 && ~isnan(row.total_power_w)
        sysMonitor.cum_energy_j = sysMonitor.cum_energy_j + row.total_power_w * dt;
    end
    
    row.cum_energy_j = sysMonitor.cum_energy_j;
end

%% ========================================================================
% MONITORING PANEL
% =========================================================================
function mon = createMonitoringPanel(variantName, imageSize)
    % Create live monitoring figure
    mon = struct();
    mon.variantName = variantName;
    mon.fig = figure('Name', sprintf('Training: %s', variantName), ...
                     'NumberTitle', 'off', 'Position', [100 100 1200 800]);
    
    % Create subplots
    mon.ax_loss = subplot(3, 3, [1 2]);
    title('Loss Components'); xlabel('Iteration'); ylabel('Loss'); grid on; hold on;
    mon.line_total = plot(NaN, NaN, 'k-', 'LineWidth', 2, 'DisplayName', 'Total');
    mon.line_rec = plot(NaN, NaN, 'b--', 'DisplayName', 'Recon');
    mon.line_geo = plot(NaN, NaN, 'r--', 'DisplayName', 'Geo');
    mon.line_phys = plot(NaN, NaN, 'g--', 'DisplayName', 'Phys');
    legend('Location', 'best');
    
    mon.ax_power = subplot(3, 3, 3);
    title('Power Consumption'); xlabel('Time (s)'); ylabel('Power (W)'); grid on; hold on;
    mon.line_power = plot(NaN, NaN, 'r-', 'LineWidth', 1.5);
    
    mon.ax_resources = subplot(3, 3, [4 5]);
    title('System Resources'); xlabel('Time (s)'); ylabel('Usage (%)'); grid on; hold on;
    mon.line_cpu = plot(NaN, NaN, 'b-', 'DisplayName', 'CPU');
    mon.line_gpu = plot(NaN, NaN, 'r-', 'DisplayName', 'GPU');
    mon.line_ram = plot(NaN, NaN, 'g-', 'DisplayName', 'RAM');
    legend('Location', 'best');
    
    mon.ax_lr = subplot(3, 3, 6);
    title('Learning Rate'); xlabel('Iteration'); ylabel('LR'); grid on; hold on;
    mon.line_lr = semilogy(NaN, NaN, 'k-', 'LineWidth', 1.5);
    
    mon.ax_energy = subplot(3, 3, [7 8]);
    title('Cumulative Energy'); xlabel('Time (s)'); ylabel('Energy (Wh)'); grid on; hold on;
    mon.line_energy = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
    
    mon.ax_R = subplot(3, 3, 9);
    title('Geometry Parameter R'); xlabel('Iteration'); ylabel('R'); grid on; hold on;
    mon.line_R = plot(NaN, NaN, 'k-', 'LineWidth', 1.5);
    
    mon.previewEvery = 50;  % preview interval (iterations)
    drawnow;  % render figure
end

function updateMonitoringPanel(mon, iteration, metricsLog, trace, Rparam)
    % Update monitoring panel with latest data
    
    % Update loss plot
    set(mon.line_total, 'XData', 1:numel(trace.loss), 'YData', trace.loss);
    set(mon.line_rec, 'XData', 1:numel(trace.recon), 'YData', trace.recon);
    set(mon.line_geo, 'XData', 1:numel(trace.geo), 'YData', trace.geo);
    set(mon.line_phys, 'XData', 1:numel(trace.phys), 'YData', trace.phys);
    
    % Update power plot
    set(mon.line_power, 'XData', metricsLog.elapsed_s, 'YData', metricsLog.total_power_w);
    
    % Update resources plot
    set(mon.line_cpu, 'XData', metricsLog.elapsed_s, 'YData', metricsLog.cpu_pct);
    set(mon.line_gpu, 'XData', metricsLog.elapsed_s, 'YData', metricsLog.gpu_util_pct);
    set(mon.line_ram, 'XData', metricsLog.elapsed_s, 'YData', metricsLog.ram_pct);
    
    % Update LR plot
    set(mon.line_lr, 'XData', 1:height(metricsLog), 'YData', metricsLog.lr);
    
    % Update energy plot
    set(mon.line_energy, 'XData', metricsLog.elapsed_s, 'YData', metricsLog.cum_energy_j / 3600);
    
    % Update R plot
    set(mon.line_R, 'XData', 1:numel(trace.R), 'YData', trace.R);
    
    drawnow limitrate;  % refresh without slowing training
end

%% ========================================================================
% PROCESS MONITORING (PCA/PLS/KPCA/KPLS) - IMPLEMENTATIONS CONTINUE...
% =========================================================================
% NOTE: Due to length constraints, PM functions will be in next section
% See unified_geomAE_PM_functions.m

%% ========================================================================
% UTILITY FUNCTIONS
% =========================================================================
% Copy fields of s2 into s1 (s2 overwrites)
function s = mergeStructs(s1, s2)
    s = s1;
    fn = fieldnames(s2);
    for i = 1:numel(fn)
        s.(fn{i}) = s2.(fn{i});
    end
end

% Save figure only if it still exists
function saveFigure(fig, filename)
    if ~isempty(fig) && isvalid(fig)
        saveas(fig, filename);
    end
end


%% ========================================================================
% SHAP LOCAL EXPLANATION
% =========================================================================
function shapLocal = computeSHAPLocal(X_sample, netE, netD, spec, Rparam, nSamples)
    % Compute SHAP values for a single sample (local explanation)
    % Uses Kernel SHAP approximation
    %
    % Inputs:
    %   X_sample: Single image sample [H x W x C]
    %   netE, netD: Encoder and decoder networks
    %   spec: Variant specification
    %   Rparam: Geometry parameter
    %   nSamples: Number of coalition samples (default: 100)
    
    if nargin < 6
        nSamples = 100;
    end
    
    fprintf('        Computing SHAP local (sample-level)...\n');
    
    % Convert to dlarray
    X_sample = dlarray(single(X_sample), 'SSCB');
    
    % Get baseline (zero image)
    X_baseline = dlarray(zeros(size(X_sample), 'like', X_sample), 'SSCB');
    
    % Forward pass for sample
    Z_sample = forward(netE, X_sample);
    Zg_sample = spec.map_to_manifold(Z_sample, Rparam);
    Xhat_sample = forward(netD, Zg_sample);
    
    % Forward pass for baseline
    Z_baseline = forward(netE, X_baseline);
    Zg_baseline = spec.map_to_manifold(Z_baseline, Rparam);
    Xhat_baseline = forward(netD, Zg_baseline);
    
    % Prediction function (reconstruction error)
    f_sample = double(gather(extractdata(mse(Xhat_sample, X_sample))));
    f_baseline = double(gather(extractdata(mse(Xhat_baseline, X_baseline))));
    
    % Create superpixels (regions) for perturbation
    sizeX = size(extractdata(X_sample));
    H_cpu = sizeX(1);  % height
    W_cpu = sizeX(2);  % width
    C_cpu = sizeX(3);  % channels
    nRegions = 16; % 4x4 grid
    regionSize_H = floor(H_cpu / sqrt(nRegions));  % region height
    regionSize_W = floor(W_cpu / sqrt(nRegions));  % region width
    
    % Generate coalitions (binary masks for each region)
    nFeatures = nRegions;
    coalitions = rand(nSamples, nFeatures) > 0.5;
    coalitions = [zeros(1, nFeatures); ones(1, nFeatures); coalitions]; % Add empty and full
    
    % Evaluate model on coalitions
    f_values = zeros(size(coalitions, 1), 1);
    X_sample_data = extractdata(X_sample);
    X_baseline_data = extractdata(X_baseline);
    
    for i = 1:size(coalitions, 1)
        % Create perturbed image
        X_pert = dlarray(extractdata(X_baseline), 'SSCB');
        mask = coalitions(i, :);
        
        for r = 1:nFeatures
            row_idx = floor((r-1) / sqrt(nRegions));  % region row in the grid
            col_idx = mod(r-1, sqrt(nRegions));  % region column in the grid
            
            % Pixel limits of region r
            h_start = row_idx * regionSize_H + 1;
            h_end = min((row_idx + 1) * regionSize_H, H_cpu);
            w_start = col_idx * regionSize_W + 1;
            w_end = min((col_idx + 1) * regionSize_W, W_cpu);
            
            if mask(r)  % region in coalition: copy original pixels
                X_pert_data = extractdata(X_pert);
                X_sample_data = extractdata(X_sample);
                X_pert_data(h_start:h_end, w_start:w_end, :, :) = ...
                    X_sample_data(h_start:h_end, w_start:w_end, :, :);
                X_pert = dlarray(X_pert_data, 'SSCB');
            end
        end
        
        % Forward pass
        Z_pert = forward(netE, X_pert);
        Zg_pert = spec.map_to_manifold(Z_pert, Rparam);
        Xhat_pert = forward(netD, Zg_pert);
        f_values(i) = double(gather(extractdata(mse(Xhat_pert, X_sample))));  % error vs original sample
    end
    
    % Kernel SHAP: solve weighted least squares
    weights = kernelSHAPWeights(coalitions);
    phi = solveSHAP(coalitions, f_values, weights);
    
    % Reshape SHAP values to image
    shapMap = zeros(H_cpu, W_cpu, 'single');
    for r = 1:nFeatures
        row_idx = floor((r-1) / sqrt(nRegions));
        col_idx = mod(r-1, sqrt(nRegions));
        
        h_start = row_idx * regionSize_H + 1;
        h_end = min((row_idx + 1) * regionSize_H, H_cpu);
        w_start = col_idx * regionSize_W + 1;
        w_end = min((col_idx + 1) * regionSize_W, W_cpu);
        
        shapMap(h_start:h_end, w_start:w_end) = phi(r);
    end
    
    % Visualization
    fig = figure('Name', 'SHAP Local', 'Position', [100 100 1200 400]);
    
    subplot(1, 3, 1);
    imshow(gather(extractdata(X_sample(:,:,1,1))), []);
    title('Original Image (Channel 1)');
    
    subplot(1, 3, 2);
    imagesc(shapMap);
    colorbar;
    title('SHAP Attribution Map');
    axis image;
    
    subplot(1, 3, 3);
    bar(phi);
    xlabel('Region');
    ylabel('SHAP Value');
    title('SHAP Values per Region');
    grid on;
    
    % Results
    shapLocal = struct();
    shapLocal.phi = phi;
    shapLocal.shapMap = shapMap;
    shapLocal.f_sample = f_sample;
    shapLocal.f_baseline = f_baseline;
    shapLocal.coalitions = coalitions;
    shapLocal.f_values = f_values;
    shapLocal.fig = fig;
    
    fprintf('        SHAP local complete.\n');
end

%% ========================================================================
% SHAP GLOBAL EXPLANATION
% =========================================================================
function shapGlobal = computeSHAPGlobal(X_samples, netE, netD, spec, Rparam, nSamples)
    % Compute global SHAP values (aggregated across multiple samples)
    %
    % Inputs:
    %   X_samples: Multiple samples [H x W x C x N]
    %   netE, netD: Encoder and decoder networks
    %   spec: Variant specification
    %   Rparam: Geometry parameter
    %   nSamples: Number of samples to analyze (default: min(50, N))
    
    fprintf('        Computing SHAP global (population-level)...\n');
    
    N = size(X_samples, 4);
    if nargin < 6
        nSamples = min(50, N);
    else
        nSamples = min(nSamples, N);
    end
    
    % Compute SHAP for each sample
    allPhi = [];
    for i = 1:nSamples
        if mod(i, 10) == 0
            fprintf('          Sample %d/%d\n', i, nSamples);
        end
        
        X_sample = X_samples(:,:,:,i);
        shapLocal = computeSHAPLocal(X_sample, netE, netD, spec, Rparam, 50);
        allPhi = [allPhi; shapLocal.phi']; %#ok<AGROW>
        close(shapLocal.fig); % Close individual figures
    end
    
    % Aggregate: mean absolute SHAP values
    meanAbsShap = mean(abs(allPhi), 1);
    stdShap = std(allPhi, 0, 1);
    
    % Visualization
    fig = figure('Name', 'SHAP Global', 'Position', [100 100 1200 400]);
    
    subplot(1, 2, 1);
    bar(meanAbsShap);
    hold on;
    errorbar(1:length(meanAbsShap), meanAbsShap, stdShap, 'k.', 'LineWidth', 1.5);
    xlabel('Region');
    ylabel('Mean |SHAP Value|');
    title('Global Feature Importance');
    grid on;
    
    subplot(1, 2, 2);
    imagesc(allPhi);
    colorbar;
    xlabel('Region');
    ylabel('Sample');
    title('SHAP Values Across Samples');
    
    % Results
    shapGlobal = struct();
    shapGlobal.allPhi = allPhi;
    shapGlobal.meanAbsShap = meanAbsShap;
    shapGlobal.stdShap = stdShap;
    shapGlobal.fig = fig;
    
    fprintf('        SHAP global complete.\n');
end

%% ========================================================================
% LIME EXPLANATION
% =========================================================================
function limeExplain = computeLIME(X_sample, netE, netD, spec, Rparam, nSamples)
    % Local Interpretable Model-agnostic Explanations (LIME)
    %
    % Inputs:
    %   X_sample: Single image sample [H x W x C]
    %   netE, netD: Encoder and decoder networks
    %   spec: Variant specification
    %   Rparam: Geometry parameter
    %   nSamples: Number of perturbations (default: 100)
    
    if nargin < 6
        nSamples = 100;
    end
    
    fprintf('        Computing LIME...\n');
    
    % Convert to dlarray
    X_sample = dlarray(single(X_sample), 'SSCB');
    sizeX = size(extractdata(X_sample));
    H_cpu = sizeX(1);
    W_cpu = sizeX(2);
    C_cpu = sizeX(3);
    
    % Original prediction
    Z_sample = forward(netE, X_sample);
    Zg_sample = spec.map_to_manifold(Z_sample, Rparam);
    Xhat_sample = forward(netD, Zg_sample);
    f_original = double(gather(extractdata(mse(Xhat_sample, X_sample))));
    
    % Create superpixels
    nRegions = 16;
    regionSize_H = floor(H_cpu / sqrt(nRegions));
    regionSize_W = floor(W_cpu / sqrt(nRegions));
    
    % Generate perturbations (uniform sampling around sample)
    X_perturbed = zeros(H_cpu, W_cpu, C_cpu, nSamples, 'single');
    perturbVecs = zeros(nSamples, nRegions);
    f_perturbed = zeros(nSamples, 1);
    distances = zeros(nSamples, 1);
    
    X_cpu = gather(extractdata(X_sample));
    
    for i = 1:nSamples
        % Random binary mask
        mask = rand(1, nRegions) > 0.3; % 70% features on average
        perturbVecs(i, :) = mask;
        
        % Apply mask
        X_pert = zeros(size(X_cpu), 'like', X_cpu);
        for r = 1:nRegions
            row_idx = floor((r-1) / sqrt(nRegions));
            col_idx = mod(r-1, sqrt(nRegions));
            
            h_start = row_idx * regionSize_H + 1;
            h_end = min((row_idx + 1) * regionSize_H, H_cpu);
            w_start = col_idx * regionSize_W + 1;
            w_end = min((col_idx + 1) * regionSize_W, W_cpu);
            
            if mask(r)
                X_pert(h_start:h_end, w_start:w_end, :) = ...
                    X_cpu(h_start:h_end, w_start:w_end, :);
            else
                % Add noise
                X_pert(h_start:h_end, w_start:w_end, :) = ...
                    X_cpu(h_start:h_end, w_start:w_end, :) + ...
                    0.1*randn(h_end-h_start+1, w_end-w_start+1, C_cpu, 'single');
            end
        end
        
        X_perturbed(:,:,:,i) = X_pert;
        
        % Evaluate
        X_pert_dl = dlarray(X_pert, 'SSCB');
        Z_pert = forward(netE, X_pert_dl);
        Zg_pert = spec.map_to_manifold(Z_pert, Rparam);
        Xhat_pert = forward(netD, Zg_pert);
        f_perturbed(i) = double(gather(extractdata(mse(Xhat_pert, X_sample))));
        
        % Distance from original (cosine distance in feature space)
        distances(i) = 1 - sum(mask) / nRegions;
    end
    
    % LIME: Fit weighted linear model
    % Weight by proximity (exponential kernel)
    kernel_width = 0.25;
    weights = exp(-distances.^2 / kernel_width^2);
    
    % Weighted least squares: f = beta * x
    X_weighted = perturbVecs .* sqrt(weights);
    y_weighted = f_perturbed .* sqrt(weights);
    
    beta = (X_weighted' * X_weighted + 1e-6*eye(nRegions)) \ (X_weighted' * y_weighted);  % ridge-regularized weighted least squares
    
    % Reshape coefficients to map
    limeMap = zeros(H_cpu, W_cpu, 'single');
    for r = 1:nRegions
        row_idx = floor((r-1) / sqrt(nRegions));
        col_idx = mod(r-1, sqrt(nRegions));
        
        h_start = row_idx * regionSize_H + 1;
        h_end = min((row_idx + 1) * regionSize_H, H_cpu);
        w_start = col_idx * regionSize_W + 1;
        w_end = min((col_idx + 1) * regionSize_W, W_cpu);
        
        limeMap(h_start:h_end, w_start:w_end) = beta(r);
    end
    
    % Visualization
    fig = figure('Name', 'LIME', 'Position', [100 100 1200 400]);
    
    subplot(1, 3, 1);
    imshow(X_cpu(:,:,1), []);
    title('Original Image (Channel 1)');
    
    subplot(1, 3, 2);
    imagesc(limeMap);
    colorbar;
    title('LIME Explanation Map');
    axis image;
    
    subplot(1, 3, 3);
    bar(beta);
    xlabel('Region');
    ylabel('LIME Coefficient');
    title('LIME Feature Importance');
    grid on;
    
    % Results
    limeExplain = struct();
    limeExplain.beta = beta;
    limeExplain.limeMap = limeMap;
    limeExplain.f_original = f_original;
    limeExplain.perturbVecs = perturbVecs;
    limeExplain.f_perturbed = f_perturbed;
    limeExplain.weights = weights;
    limeExplain.fig = fig;
    
    fprintf('        LIME complete.\n');
end

%% ========================================================================
% MONTE CARLO CLASSIFICATION ANALYSIS
% =========================================================================
function mcStats = runMonteCarloClassification(XTest, netE, netD, spec, Rparam, nTrials)
    % Run Monte Carlo trials for classification robustness analysis
    % Adds noise and evaluates classification metrics
    %
    % Inputs:
    %   XTest: Test images [H x W x C x N]
    %   netE, netD: Encoder and decoder networks
    %   spec: Variant specification
    %   Rparam: Geometry parameter
    %   nTrials: Number of Monte Carlo trials
    %
    % Outputs:
    %   mcStats: Statistics (mean, std) of classification metrics
    
    fprintf('      Running Monte Carlo (%d trials)...\n', nTrials);
    
    N = size(XTest, 4);
    
    % Initialize storage
    accuracy_trials = zeros(nTrials, 1);
    precision_trials = zeros(nTrials, 1);
    recall_trials = zeros(nTrials, 1);
    f1_trials = zeros(nTrials, 1);
    mse_trials = zeros(nTrials, 1);
    mae_trials = zeros(nTrials, 1);
    
    % Baseline (no noise)
    fprintf('        Computing baseline...\n');
    dlX_clean = dlarray(single(XTest), 'SSCB');
    Z_clean = forward(netE, dlX_clean);
    Zg_clean = spec.map_to_manifold(Z_clean, Rparam);
    Xhat_clean = forward(netD, Zg_clean);
    
    % Threshold for classification (based on reconstruction error)
    reconError_clean = squeeze(sum((XTest - gather(extractdata(Xhat_clean))).^2, [1 2 3]));
    threshold = prctile(reconError_clean, 95); % 95th percentile
    
    % Monte Carlo trials
    for trial = 1:nTrials
        if mod(trial, 10) == 0
            fprintf('        Trial %d/%d\n', trial, nTrials);
        end
        
        % Add noise
        noiseLevel = 0.05 + 0.05*rand; % Random noise level [0.05, 0.10]
        X_noisy = XTest + noiseLevel * randn(size(XTest), 'single');
        
        % Forward pass
        dlX_noisy = dlarray(X_noisy, 'SSCB');
        Z_noisy = forward(netE, dlX_noisy);
        Zg_noisy = spec.map_to_manifold(Z_noisy, Rparam);
        Xhat_noisy = forward(netD, Zg_noisy);
        
        % Reconstruction errors
        reconError_noisy = squeeze(sum((X_noisy - gather(extractdata(Xhat_noisy))).^2, [1 2 3]));
        
        % Classification (normal if recon error < threshold)
        y_pred = reconError_noisy < threshold;
        y_true = reconError_clean < threshold; % Ground truth from clean
        
        % Compute metrics
        TP = sum(y_pred & y_true);  % true positives
        TN = sum(~y_pred & ~y_true);  % true negatives
        FP = sum(y_pred & ~y_true);  % false positives
        FN = sum(~y_pred & y_true);  % false negatives
        
        accuracy_trials(trial) = (TP + TN) / N;
        precision_trials(trial) = TP / (TP + FP + 1e-9);
        recall_trials(trial) = TP / (TP + FN + 1e-9);
        f1_trials(trial) = 2 * precision_trials(trial) * recall_trials(trial) / ...
                               (precision_trials(trial) + recall_trials(trial) + 1e-9);
        
        % Reconstruction metrics
        err = X_noisy - gather(extractdata(Xhat_noisy));
        mse_trials(trial) = mean(err(:).^2);
        mae_trials(trial) = mean(abs(err(:)));
    end
    
    % Aggregate statistics
    mcStats = struct();
    % Mean, std and per-trial values of each metric
    mcStats.nTrials = nTrials;
    mcStats.accuracy_mean = mean(accuracy_trials);
    mcStats.accuracy_std = std(accuracy_trials);
    mcStats.accuracy_trials = accuracy_trials;
    mcStats.precision_mean = mean(precision_trials);
    mcStats.precision_std = std(precision_trials);
    mcStats.precision_trials = precision_trials;
    mcStats.recall_mean = mean(recall_trials);
    mcStats.recall_std = std(recall_trials);
    mcStats.recall_trials = recall_trials;
    mcStats.f1_mean = mean(f1_trials);
    mcStats.f1_std = std(f1_trials);
    mcStats.f1_trials = f1_trials;
    mcStats.mse_mean = mean(mse_trials);
    mcStats.mse_std = std(mse_trials);
    mcStats.mse_trials = mse_trials;
    mcStats.mae_mean = mean(mae_trials);
    mcStats.mae_std = std(mae_trials);
    mcStats.mae_trials = mae_trials;
    
    fprintf('      Monte Carlo complete.\n');
end

%% ========================================================================
% HELPER FUNCTIONS
% =========================================================================
function weights = kernelSHAPWeights(coalitions)
    % Compute kernel weights for SHAP
    % Follows the Shapley kernel from Lundberg & Lee 2017
    
    M = size(coalitions, 2); % Number of features
    z = sum(coalitions, 2); % Number of present features
    
    %weights = (M - 1) ./ (nchoosek(M, z) .* z .* (M - z) + 1e-9);
    weights = zeros(size(z));
    for i = 1:length(z)
        if z(i) == 0 || z(i) == M
            weights(i) = 1e9; % Very high weight for empty and full coalitions
        else
            weights(i) = (M - 1) / (nchoosek(M, z(i)) * z(i) * (M - z(i)) + 1e-9);
        end
    end
    weights(z == 0 | z == M) = 1e6; % Large weight for empty and full
    
    weights = weights / sum(weights); % Normalize
end

function phi = solveSHAP(coalitions, f_values, weights)
    % Solve for SHAP values using weighted least squares
    %
    % min_phi sum_i weights(i) * (f(i) - sum_j phi_j * z_ij)^2
    
    % Weighted least squares
    W = diag(sqrt(weights));
    A = W * coalitions;
    b = W * f_values;
    
    % Solve: A * phi = b
    phi = (A' * A + 1e-6 * eye(size(A, 2))) \ (A' * b);
end

%% ========================================================================
% FINAL KPI COMPUTATION
% =========================================================================
function finalKPIs = computeFinalKPIs(results, pmResults, mcResults, allGeometricKPIs)
    % Consolidate all KPIs into final summary tables
    
    fprintf('\n  Computing final KPI consolidation...\n');
    
    variantNames = fieldnames(results);  % variant names
    nVariants = numel(variantNames);
    
    % Initialize summary table
    summary = table();
    
    for v = 1:nVariants
        variantName = variantNames{v};
        res = results.(variantName);
        
        row = struct();
        row.variant = string(variantName);
        
        % Training metrics
        row.best_loss = res.bestLoss;
        row.best_iter = res.bestIter;
        row.train_time_s = res.totalTime;
        
        % Energy consumption
        if ~isempty(res.metricsLog)
            row.total_energy_j = res.metricsLog.cum_energy_j(end);
            row.total_energy_wh = row.total_energy_j / 3600;
            row.avg_power_w = mean(res.metricsLog.total_power_w, 'omitnan');
        else
            row.total_energy_j = NaN;
            row.total_energy_wh = NaN;
            row.avg_power_w = NaN;
        end
        
        % Test reconstruction metrics
        row.test_mse = res.testResults.MSE;
        row.test_mae = res.testResults.MAE;
        row.test_ssim = res.testResults.SSIM;
        
        % Geometric metrics
        row.geo_dist_mean = res.geoKPIs.geo_dist_mean;
        row.geo_dist_std = res.geoKPIs.geo_dist_std;
        row.geo_dist_max = res.geoKPIs.geo_dist_max;
        
        % Physics metrics
        row.phys_dir_mse = res.geoKPIs.phys_dir_mse;
        row.phys_ind_mse = res.geoKPIs.phys_ind_mse;
        
        % Process monitoring metrics
        if isfield(pmResults, variantName)
            pm = pmResults.(variantName);
            row.pm_pca_anomaly_rate = pm.PCA.anomaly_rate;
            row.pm_pca_q2 = pm.PCA.Q2;
            
            if isfield(pm, 'PLS')
                row.pm_pls_anomaly_rate = pm.PLS.anomaly_rate;
                row.pm_pls_q2 = pm.PLS.Q2;
            else
                row.pm_pls_anomaly_rate = NaN;
                row.pm_pls_q2 = NaN;
            end
            
            row.pm_kpca_anomaly_rate = pm.KPCA.anomaly_rate;
            row.pm_kpca_q2 = pm.KPCA.Q2;
            
            if isfield(pm, 'KPLS')
                row.pm_kpls_anomaly_rate = pm.KPLS.anomaly_rate;
                row.pm_kpls_q2 = pm.KPLS.Q2;
            else
                row.pm_kpls_anomaly_rate = NaN;
                row.pm_kpls_q2 = NaN;
            end
        else
            row.pm_pca_anomaly_rate = NaN;
            row.pm_pca_q2 = NaN;
            row.pm_pls_anomaly_rate = NaN;
            row.pm_pls_q2 = NaN;
            row.pm_kpca_anomaly_rate = NaN;
            row.pm_kpca_q2 = NaN;
            row.pm_kpls_anomaly_rate = NaN;
            row.pm_kpls_q2 = NaN;
        end
        
        % Monte Carlo metrics
        if isfield(mcResults, variantName)
            mc = mcResults.(variantName);
            row.mc_accuracy_mean = mc.accuracy_mean;
            row.mc_accuracy_std = mc.accuracy_std;
            row.mc_precision_mean = mc.precision_mean;
            row.mc_precision_std = mc.precision_std;
            row.mc_recall_mean = mc.recall_mean;
            row.mc_recall_std = mc.recall_std;
            row.mc_f1_mean = mc.f1_mean;
            row.mc_f1_std = mc.f1_std;
        else
            row.mc_accuracy_mean = NaN;
            row.mc_accuracy_std = NaN;
            row.mc_precision_mean = NaN;
            row.mc_precision_std = NaN;
            row.mc_recall_mean = NaN;
            row.mc_recall_std = NaN;
            row.mc_f1_mean = NaN;
            row.mc_f1_std = NaN;
        end
        
        summary = [summary; struct2table(row)]; %#ok<AGROW>
    end
    
    % Final output
    finalKPIs = struct();
    finalKPIs.summary = summary;
    finalKPIs.geometric = allGeometricKPIs;
    
    fprintf('  Final KPI consolidation complete.\n');
end

%% ========================================================================
% FINAL VISUALIZATION
% =========================================================================
function generateFinalPlots(finalKPIs, allGeometricKPIs, results)
    % Generate comprehensive comparison plots
    
    fprintf('  Generating final comparison plots...\n');
    
    T = finalKPIs.summary;
    
    % Plot 1: Energy vs. Reconstruction Error
    fig1 = figure('Name', 'Energy vs Reconstruction', 'Position', [100 100 800 600]);
    scatter(T.total_energy_wh, T.test_mse, 100, 'filled');
    xlabel('Total Energy (Wh)');
    ylabel('Test MSE');
    title('Energy Consumption vs. Reconstruction Error');
    grid on;
    % Label each point with its variant
    text(T.total_energy_wh, T.test_mse, T.variant, ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    saveas(fig1, 'results/figures/energy_vs_recon.png');  % save PNG
    
    % Plot 2: Training Time Comparison
    fig2 = figure('Name', 'Training Time', 'Position', [100 100 1000 600]);
    bar(categorical(T.variant), T.train_time_s);
    ylabel('Training Time (s)');
    title('Training Time by Variant');
    xtickangle(45);
    grid on;
    saveas(fig2, 'results/figures/training_time_comparison.png');
    
    % Plot 3: Monte Carlo Classification Performance
    fig3 = figure('Name', 'MC Classification', 'Position', [100 100 1200 500]);
    subplot(1, 2, 1);
    bar(categorical(T.variant), T.mc_accuracy_mean);
    hold on;
    errorbar(1:height(T), T.mc_accuracy_mean, T.mc_accuracy_std, 'k.', 'LineWidth', 1.5);
    ylabel('Accuracy');
    title('Monte Carlo: Mean Accuracy ± Std');
    xtickangle(45);
    grid on;
    
    subplot(1, 2, 2);
    bar(categorical(T.variant), T.mc_f1_mean);
    hold on;
    errorbar(1:height(T), T.mc_f1_mean, T.mc_f1_std, 'k.', 'LineWidth', 1.5);
    ylabel('F1-Score');
    title('Monte Carlo: Mean F1-Score ± Std');
    xtickangle(45);
    grid on;
    saveas(fig3, 'results/figures/monte_carlo_classification.png');
    
    % Plot 4: Geometric Distance Distribution
    fig4 = figure('Name', 'Geometric Distances', 'Position', [100 100 1000 600]);
    bar(categorical(T.variant), T.geo_dist_mean);
    hold on;
    errorbar(1:height(T), T.geo_dist_mean, T.geo_dist_std, 'r.', 'LineWidth', 1.5);
    ylabel('Geodesic Distance');
    title('Mean Geometric Distance in Latent Space');
    xtickangle(45);
    grid on;
    saveas(fig4, 'results/figures/geometric_distances.png');
    
    % Plot 5: Process Monitoring Q2 Comparison
    fig5 = figure('Name', 'PM Q2 Comparison', 'Position', [100 100 1200 600]);
    X = categorical(T.variant);  % x axis: variants
    Y = [T.pm_pca_q2, T.pm_pls_q2, T.pm_kpca_q2, T.pm_kpls_q2];  % one bar per method
    bar(X, Y);
    ylabel('Q^2 Coefficient');
    title('Process Monitoring: Q^2 Comparison');
    legend({'PCA', 'PLS', 'KPCA', 'KPLS'}, 'Location', 'best');
    xtickangle(45);
    grid on;
    saveas(fig5, 'results/figures/pm_q2_comparison.png');
    
    fprintf('  Plots saved to results/figures/\n');
end

% function [MSEm,MAEm,dSSIMm]=kpi_error_multichannel_images_geom(Im1,Im2)
%     % Convert images to double for calculations
%     ImA = single(Im1);
%     ImB = single(Im2);
%     MSEm = mean((ImA(:) - ImB(:)).^2);
%     MAEm = mean(abs(ImA(:) - ImB(:)));
%     % Structural Similarity Index (SSIM)
%     % The `ssim` function is part of the Image Processing Toolbox
%     [SSIMm, ~] = ssim(ImB,ImA);
%     dSSIMm = 1 - SSIMm;%psnr(ImB,ImA);
% 
% end

function [kpis_original, kpis_geometric, infoKPIs] = kpi_error_multichannel_images_geom(XTest, XTesthat, ZTest, ZTesthat, spec, Rparam)
% KPI_ERROR_MULTICHANNEL_IMAGES_GEOM Computes comprehensive KPIs for 
% geometric autoencoders in the original and latent space
%
% Inputs:
%   XTest: Original images [H x W x C x N]
%   XTesthat: Reconstructed images [H x W x C x N]
%   ZTest: Original embeddings [d x N]
%   ZTesthat: Reconstructed embeddings [d x N]
%   spec: Geometric model specification (struct with fields:
%         - map_to_manifold: projection function
%         - geo_distance: geodesic distance function
%         - latentDim: latent space dimension)
%   Rparam: Curvature parameter (scalar dlarray or numeric)
%
% Outputs:
%   kpis_original: Struct with KPIs in image space
%   kpis_geometric: Struct with KPIs in the geometric latent space
%   infoKPIs: Struct with detailed per-sample information

    %% ===================================================================
    %% PART 1: KPIs IN ORIGINAL SPACE (IMAGES)
    %% ===================================================================
    
    % Convert to numeric format if needed
    if isa(XTest, 'dlarray')
        XTest = extractdata(XTest);
    end
    if isa(XTesthat, 'dlarray')
        XTesthat = extractdata(XTesthat);
    end
    
    XTest = single(XTest);
    XTesthat = single(XTesthat);
    
    N = size(XTest, 4);  % number of samples
    
    % Initialize per-sample KPI arrays
    MSE_samples = zeros(N, 1);
    MAE_samples = zeros(N, 1);
    dSSIM_samples = zeros(N, 1);
    RMSEE_samples = zeros(N, 1);
    
    % Compute KPIs for each sample
    for i = 1:N
        img_orig = XTest(:,:,:,i);
        img_recon = XTesthat(:,:,:,i);
        
        % 1. Mean Squared Error (MSE)
        err = img_orig - img_recon;
        MSE_samples(i) = mean(err(:).^2);
        
        % 2. Mean Absolute Error (MAE)
        MAE_samples(i) = mean(abs(err(:)));
        
        % 3. Structural Similarity Index (SSIM) → dSSIM
        try
            % Compute SSIM per channel and average
            ssim_vals = zeros(size(img_orig, 3), 1);
            for ch = 1:size(img_orig, 3)
                ssim_vals(ch) = ssim(img_recon(:,:,ch), img_orig(:,:,ch));
            end
            SSIM_mean = mean(ssim_vals);
            dSSIM_samples(i) = 1 - SSIM_mean;
        catch
            % If SSIM fails, use an MSE-based approximation
            dSSIM_samples(i) = MSE_samples(i);
        end
        
        % 4. Root Mean Squared Logarithmic Error (RMSEE)
        epsilon = 1e-7; % Constant for numerical stability
        log_err = log(img_orig(:) + epsilon) - log(img_recon(:) + epsilon);
        RMSEE_samples(i) = sqrt(mean(log_err.^2));
    end
    
    % Aggregate statistics
    kpis_original = struct();
    kpis_original.MSE_mean = mean(MSE_samples);
    kpis_original.MSE_std = std(MSE_samples);
    kpis_original.MSE_samples = MSE_samples;
    
    kpis_original.MAE_mean = mean(MAE_samples);
    kpis_original.MAE_std = std(MAE_samples);
    kpis_original.MAE_samples = MAE_samples;
    
    kpis_original.dSSIM_mean = mean(dSSIM_samples);
    kpis_original.dSSIM_std = std(dSSIM_samples);
    kpis_original.dSSIM_samples = dSSIM_samples;
    
    kpis_original.RMSEE_mean = mean(RMSEE_samples);
    kpis_original.RMSEE_std = std(RMSEE_samples);
    kpis_original.RMSEE_samples = RMSEE_samples;
    
    %% ===================================================================
    %% PART 2: KPIs IN GEOMETRIC SPACE (EMBEDDINGS)
    %% ===================================================================
    
    % Convert embeddings to numeric format
    if isa(ZTest, 'dlarray')
        ZTest = extractdata(ZTest);
    end
    if isa(ZTesthat, 'dlarray')
        ZTesthat = extractdata(ZTesthat);
    end
    if isa(Rparam, 'dlarray')
        Rparam = extractdata(Rparam);
    end
    
    ZTest = double(ZTest);
    ZTesthat = double(ZTesthat);
    Rparam = double(Rparam);
    
    [d, N] = size(ZTest);  % latent dim and number of samples
    
    % Initialize geometric KPI struct
    kpis_geometric = struct();
    
    % ===================================================================
    % 1. DISTORTION: global distance preservation
    % ===================================================================
    
    % Sample pairs for efficiency (use all samples if N is small)
    if N <= 100
        pairs_idx = nchoosek(1:N, 2);
    else
        % Sample 1000 random pairs
        n_pairs = min(1000, nchoosek(N, 2));
        pairs_idx = zeros(n_pairs, 2);
        for p = 1:n_pairs
            pairs_idx(p,:) = randsample(N, 2);  % random pair of distinct indices
        end
    end
    
    n_pairs = size(pairs_idx, 1);
    distortions = zeros(n_pairs, 1);
    
    for p = 1:n_pairs
        i = pairs_idx(p, 1);
        j = pairs_idx(p, 2);
        
        % Euclidean distance in original space
        d_euclidean = norm(ZTest(:,i) - ZTest(:,j));
        
        % Geodesic distance in geometric space
        zi = ZTest(:,i);
        zj = ZTest(:,j);
        
        % Apply geometric transform if needed
        zi_geom = applyGeometricTransform(zi, spec, Rparam);
        zj_geom = applyGeometricTransform(zj, spec, Rparam);
        
        % Compute geodesic distance
        d_geodesic = computeGeodesicDistance(zi_geom, zj_geom, spec, Rparam);
        
        % Distortion: |d_geo/d_euc - 1|
        if d_euclidean > 1e-9
            distortions(p) = abs(d_geodesic / d_euclidean - 1);
        else
            distortions(p) = 0;
        end
    end
    
    kpis_geometric.Distortion_mean = mean(distortions);
    kpis_geometric.Distortion_std = std(distortions);
    kpis_geometric.Distortion_samples = distortions;
    
    % ===================================================================
    % 2. MEAN RANK (MR): neighborhood preservation
    % ===================================================================
    
    k_neighbors = min(10, N-1); % Number of neighbors to consider
    ranks = zeros(N, 1);
    
    for i = 1:N
        % Distances in original space (Euclidean)
        dists_orig = sqrt(sum((ZTest - ZTest(:,i)).^2, 1));
        [~, idx_orig] = sort(dists_orig);
        idx_orig = idx_orig(2:k_neighbors+1); % Exclude the point itself
        
        % Distances in geometric space
        zi_geom = applyGeometricTransform(ZTest(:,i), spec, Rparam);
        dists_geom = zeros(1, N);
        for j = 1:N
            zj_geom = applyGeometricTransform(ZTest(:,j), spec, Rparam);
            dists_geom(j) = computeGeodesicDistance(zi_geom, zj_geom, spec, Rparam);
        end
        [~, idx_geom] = sort(dists_geom);
        
        % Compute mean rank of the original k-neighbors
        rank_sum = 0;
        for kk = 1:k_neighbors
            original_neighbor = idx_orig(kk);
            rank_in_geom = find(idx_geom == original_neighbor, 1);
            if ~isempty(rank_in_geom)
                rank_sum = rank_sum + rank_in_geom;
            else
                rank_sum = rank_sum + N; % Maximum penalty
            end
        end
        ranks(i) = rank_sum / k_neighbors;
    end
    
    kpis_geometric.MR_mean = mean(ranks);
    kpis_geometric.MR_std = std(ranks);
    kpis_geometric.MR_samples = ranks;
    
    % ===================================================================
    % 3. MEAN AVERAGE PRECISION (mAP): neighborhood precision
    % ===================================================================
    
    mAP_scores = zeros(N, 1);
    
    for i = 1:N
        % Ground truth: k-neighbors in original space
        dists_orig = sqrt(sum((ZTest - ZTest(:,i)).^2, 1));
        [~, idx_orig] = sort(dists_orig);
        neighbors_gt = idx_orig(2:k_neighbors+1);
        
        % Neighbors in geometric space
        zi_geom = applyGeometricTransform(ZTest(:,i), spec, Rparam);
        dists_geom = zeros(1, N);
        for j = 1:N
            zj_geom = applyGeometricTransform(ZTest(:,j), spec, Rparam);
            dists_geom(j) = computeGeodesicDistance(zi_geom, zj_geom, spec, Rparam);
        end
        [~, idx_geom] = sort(dists_geom);
        neighbors_pred = idx_geom(2:k_neighbors+1);
        
        % Compute Average Precision
        precision_at_k = zeros(k_neighbors, 1);
        for kk = 1:k_neighbors
            retrieved = neighbors_pred(1:kk);
            relevant = neighbors_gt(1:kk);
            tp = numel(intersect(retrieved, relevant));  % correct neighbors among the first kk
            precision_at_k(kk) = tp / kk;
        end
        mAP_scores(i) = mean(precision_at_k);  % AP of sample i
    end
    
    kpis_geometric.mAP_mean = mean(mAP_scores);
    kpis_geometric.mAP_std = std(mAP_scores);
    kpis_geometric.mAP_samples = mAP_scores;
    
    % ===================================================================
    % 4. TSAP (Tangent Space Angle Preservation)
    % ===================================================================
    
    % Compute tangent vectors in original and reconstructed space
    angleDiffs = zeros(N-1, 1);
    
    for i = 1:N-1
        % Vectors in original space (pre-transform embeddings)
        v_orig = ZTest(:,i+1) - ZTest(:,i);
        v_orig = v_orig / (norm(v_orig) + 1e-9);
        
        % Vectors in geometric space
        zi_geom = applyGeometricTransform(ZTest(:,i), spec, Rparam);
        zi1_geom = applyGeometricTransform(ZTest(:,i+1), spec, Rparam);
        
        % Tangent vector in geometric space (approximation)
        v_geom = computeTangentVector(zi_geom, zi1_geom, spec, Rparam);
        v_geom = v_geom / (norm(v_geom) + 1e-9);
        
        % Angle between tangent vectors
        cos_angle = dot(v_orig, v_geom);
        cos_angle = max(-1, min(1, cos_angle)); % Clip for stability
        angleDiffs(i) = acos(cos_angle);
    end
    
    kpis_geometric.TSAP_mean_angle_diff = mean(angleDiffs);
    kpis_geometric.TSAP_std_angle_diff = std(angleDiffs);
    kpis_geometric.TSAP_samples = angleDiffs;
    
    % ===================================================================
    % 5. LCD (Local Curvature Deviation)
    % ===================================================================
    
    % Use local PCA to estimate curvature
    k_local = min(10, N-1);
    kappa_orig = zeros(N, 1);
    kappa_geom = zeros(N, 1);
    
    for i = 1:min(N, 100) % Limit to 100 samples for efficiency
        % Neighbors in original space
        dists = sqrt(sum((ZTest - ZTest(:,i)).^2, 1));
        [~, idx] = sort(dists);
        neighbors_idx = idx(2:k_local+1);
        
        % Local PCA in original space
        neighbors_orig = ZTest(:, neighbors_idx);
        neighbors_centered = neighbors_orig - mean(neighbors_orig, 2);
        [~, S_orig, ~] = svd(neighbors_centered, 'econ');
        
        % Curvature estimated as ratio of singular values
        singular_vals = diag(S_orig);
        if length(singular_vals) >= 2
            kappa_orig(i) = singular_vals(end) / (singular_vals(1) + 1e-9);
        else
            kappa_orig(i) = 0;
        end
        
        % Local PCA in geometric space
        neighbors_geom = zeros(d, k_local);
        for kk = 1:k_local
            neighbors_geom(:,kk) = applyGeometricTransform(ZTest(:, neighbors_idx(kk)), spec, Rparam);
        end
        neighbors_geom_centered = neighbors_geom - mean(neighbors_geom, 2);
        [~, S_geom, ~] = svd(neighbors_geom_centered, 'econ');
        
        singular_vals_geom = diag(S_geom);
        if length(singular_vals_geom) >= 2
            kappa_geom(i) = singular_vals_geom(end) / (singular_vals_geom(1) + 1e-9);
        else
            kappa_geom(i) = 0;
        end
    end
    
    % Local curvature deviation
    curv_deviation = abs(kappa_orig - kappa_geom);
    
    kpis_geometric.LCD_mean = mean(curv_deviation(1:min(N,100)));
    kpis_geometric.LCD_std = std(curv_deviation(1:min(N,100)));
    kpis_geometric.LCD_kappa_orig = kappa_orig;
    kpis_geometric.LCD_kappa_geom = kappa_geom;
    kpis_geometric.LCD_samples = curv_deviation;
    
    %% ===================================================================
    %% PART 3: DETAILED INFORMATION (OPTIONAL, FOR PLOTTING)
    %% ===================================================================
    
    infoKPIs = struct();
    infoKPIs.angleDiffs = angleDiffs;
    infoKPIs.kappaOrig = kappa_orig;
    infoKPIs.kappaProj = kappa_geom;
    infoKPIs.distortions = distortions;
    infoKPIs.ranks = ranks;
    infoKPIs.mAP_scores = mAP_scores;
    
    % Optional plotting
    if nargout == 0 || nargout > 3
        visualizeKPIs(infoKPIs);
    end
    
end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function z_geom = applyGeometricTransform(z, spec, R)
    % Applies the model's specific geometric transform
    if isfield(spec, 'map_to_manifold') && ~isempty(spec.map_to_manifold)
        z_geom = spec.map_to_manifold(z, R);
    else
        % If there is no transform, return the original
        z_geom = z;
    end
    
    % Convert to numeric if dlarray
    if isa(z_geom, 'dlarray')
        z_geom = double(extractdata(z_geom));
    end
end

function d = computeGeodesicDistance(z1, z2, spec, R)
    % Computes geodesic distance for the model's geometry
    if isfield(spec, 'geo_distance') && ~isempty(spec.geo_distance)
        d_raw = spec.geo_distance(z1, z2, R);
    else
        % Euclidean distance by default
        d_raw = sqrt(sum((z1 - z2).^2));
    end
    
    % Convert to numeric
    if isa(d_raw, 'dlarray')
        d = double(extractdata(d_raw));
    else
        d = double(d_raw);
    end
    
    % Ensure positive scalar
    if numel(d) > 1
        d = d(1);
    end
    d = abs(d);
end

function v_tangent = computeTangentVector(z1, z2, spec, R)
    % Computes approximate tangent vector in geometric space
    % Approximation: finite difference in the embedding space
    
    % Convert to numeric
    if isa(z1, 'dlarray')
        z1 = double(extractdata(z1));
    end
    if isa(z2, 'dlarray')
        z2 = double(extractdata(z2));
    end
    
    % Approximate tangent vector
    v_tangent = z2 - z1;
    
    % For the Poincaré ball: project to the tangent space at z1
    if contains(func2str(spec.map_to_manifold), 'poincare', 'IgnoreCase', true)
        % Tangent space in the Poincaré model
        norm_z1_sq = sum(z1.^2);
        lambda = 2 / (1 - norm_z1_sq + 1e-9);
        v_tangent = v_tangent / lambda;
    end
    
    % For Lorentz: already in hyperboloid coordinates
    % No extra adjustment needed in this approximation
end

function visualizeKPIs(infoKPIs)
    % Plot geometric KPIs
    figure('Name', 'Geometric KPIs Analysis', 'Position', [100 100 1200 800]);
    
    % Subplot 1: distribution of angle differences (TSAP)
    subplot(2, 3, 1);
    histogram(infoKPIs.angleDiffs, 50, 'FaceColor', [0.2 0.6 0.8]);
    xlabel('Angle Difference (rad)');
    ylabel('Frequency');
    title('TSAP: Tangent Space Angle Preservation');
    grid on;
    
    % Subplot 2: distribution of local curvature
    subplot(2, 3, 2);
    histogram(abs(infoKPIs.kappaOrig - infoKPIs.kappaProj), 50, 'FaceColor', [0.8 0.4 0.2]);
    xlabel('Curvature Deviation');
    ylabel('Frequency');
    title('LCD: Local Curvature Deviation');
    grid on;
    
    % Subplot 3: Distortion
    subplot(2, 3, 3);
    histogram(infoKPIs.distortions, 50, 'FaceColor', [0.4 0.8 0.2]);
    xlabel('Distortion');
    ylabel('Frequency');
    title('Distortion: Global Distance Preservation');
    grid on;
    
    % Subplot 4: Mean Rank
    subplot(2, 3, 4);
    histogram(infoKPIs.ranks, 50, 'FaceColor', [0.8 0.2 0.6]);
    xlabel('Mean Rank');
    ylabel('Frequency');
    title('MR: Neighborhood Rank Preservation');
    grid on;
    
    % Subplot 5: mAP scores
    subplot(2, 3, 5);
    histogram(infoKPIs.mAP_scores, 50, 'FaceColor', [0.6 0.2 0.8]);
    xlabel('mAP Score');
    ylabel('Frequency');
    title('mAP: Neighborhood Precision');
    grid on;
    
    % Subplot 6: curvature scatter plot
    subplot(2, 3, 6);
    scatter(infoKPIs.kappaOrig, infoKPIs.kappaProj, 30, 'filled', 'MarkerFaceAlpha', 0.5);
    xlabel('Original Space Curvature');
    ylabel('Geometric Space Curvature');
    title('LCD: Curvature Comparison');
    grid on;
    hold on;
    plot([0 max(infoKPIs.kappaOrig)], [0 max(infoKPIs.kappaOrig)], 'r--', 'LineWidth', 2);
    legend('Samples', 'Ideal (y=x)', 'Location', 'best');
    hold off;
    
    sgtitle('Comprehensive Geometric KPI Analysis', 'FontSize', 14, 'FontWeight', 'bold');
end

% Confusion matrix and classification metrics (1 = attack)
function [accuracy,precision,recall,f1_score]=final_result_summary(reference,result)

    % Compute the confusion matrix
    confMat = confusionmat(reference, result);
    
    % Display the confusion matrix
    disp('Confusion Matrix:');
    disp(confMat);
    
    % Extract elements of the confusion matrix
    TP = confMat(2, 2); % True Positives
    TN = confMat(1, 1); % True Negatives
    FP = confMat(1, 2); % False Positives
    FN = confMat(2, 1); % False Negatives
    
    % Compute accuracy
    accuracy = (TP + TN) / sum(confMat(:));
    % fprintf('Accuracy: %.4f\n', accuracy);
    
    % Additional Metrics (optional)
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);
end

% Adaptive threshold per KPI from an EWMA control chart
function thresholds = computeAdaptiveEWMAThresholds(all_kpis, config)
lambda = config.ewmaLambda;  % EWMA smoothing factor
L = 3.0;  % control limit width (sigmas)
thresholds = struct();
kpi_names = fieldnames(all_kpis);  % KPI names

for k = 1:numel(kpi_names)
    data = all_kpis.(kpi_names{k});
    data = data(isfinite(data));  % drop NaN/Inf
    N = numel(data);
    
    % Too few samples: use a percentile threshold
    if N < 50
        thresholds.(kpi_names{k}) = prctile(data, config.percentileLevel);
        continue;
    end
    
    % EWMA recursion
    ewma = zeros(N, 1);
    ewma(1) = data(1);
    for i = 2:N
        ewma(i) = lambda * data(i) + (1 - lambda) * ewma(i-1);
    end
    
    sigma_ewma = std(data) * sqrt(lambda / (2 - lambda));  % EWMA standard deviation
    
    threshold_val = ewma(end) + L * sigma_ewma;  % upper control limit
    
    % mAP: lower is worse, so use the lower limit
    if strcmp(kpi_names{k}, 'mAP')
        threshold_val = ewma(end) - L * sigma_ewma;
    end
    
    thresholds.(kpi_names{k}) = threshold_val;
end
end
% Adaptive threshold per KPI from a CUSUM chart
function thresholds = computeAdaptiveCUSUMThresholds(all_kpis, config)
delta = config.cusumDelta;  % shift size to detect (in sigmas)
h = 5.0;  % decision limit
thresholds = struct();
kpi_names = fieldnames(all_kpis);

for k = 1:numel(kpi_names)
    data = all_kpis.(kpi_names{k});
    data = data(isfinite(data));
    N = numel(data);
    
    if N < 50
        thresholds.(kpi_names{k}) = prctile(data, config.percentileLevel);
        continue;
    end
    
    n_baseline = floor(N * 0.5);  % first half used as baseline
    mu0 = mean(data(1:n_baseline));
    sigma0 = std(data(1:n_baseline));
    
    k_ref = delta * sigma0 / 2;  % CUSUM reference value
    
    Cp = zeros(N, 1);  % upper CUSUM
    Cn = zeros(N, 1);  % lower CUSUM
    
    for i = 2:N
        Cp(i) = max(0, Cp(i-1) + (data(i) - mu0 - k_ref));
        Cn(i) = max(0, Cn(i-1) - (data(i) - mu0 + k_ref));
    end
    
    cusum_max = max(max(Cp), max(Cn));
    
    if cusum_max > h * sigma0
        threshold_val = mu0 + delta * sigma0;  % shift detected: threshold from baseline
    else
        threshold_val = prctile(data, config.percentileLevel);  % no shift: percentile threshold
    end
    
    % mAP: lower is worse
    if strcmp(kpi_names{k}, 'mAP')
        threshold_val = mu0 - delta * sigma0;
    end
    
    thresholds.(kpi_names{k}) = threshold_val;
end
end