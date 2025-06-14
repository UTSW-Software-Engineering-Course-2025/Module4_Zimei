%% ANALYSIS OF AR-HMM CALCIUM SPIKE DETECTION RESULTS
%
% This script loads the output files generated by the
% 'pipeline_CaSpEr_SWE_template.m' script and performs further analysis
% and visualization.
%
% VERSION 3: Time-series plots (Raster, Synchrony, Single-Neuron) are now
% split into two halves for better visibility and scaling.

clear; clc; close all;

%% 1. SETUP: Define Paths, Create Directories, and Load Data
% -------------------------------------------------------------------------
disp('Setting up paths and loading data...');
basePath = "D:\utsw-study\softwareE\Module_4\Module_4_Materials-main\days2-3";
outputDir_HMM = fullfile(basePath, "/output", "HMM_template");
rawDatDir = fullfile(basePath, "/raw");
analysisDir = fullfile(basePath, "/output", "/analysis");

if ~isfolder(analysisDir)
    disp(['Creating new directory for analysis results at: ', analysisDir]);
    mkdir(analysisDir);
end

if ~isfolder(outputDir_HMM)
    error("The HMM output directory does not exist. Please run the first pipeline script first.");
end

% Load data from CSV files
hmm_statemap = readmatrix(fullfile(outputDir_HMM, 'hmm_statemap.csv'));
hmm_binarymap = readmatrix(fullfile(outputDir_HMM, 'hmm_binarymap.csv'));
intercepts = readmatrix(fullfile(outputDir_HMM, 'arhmm_coef_intercept.csv'));
ar1_coeffs = readmatrix(fullfile(outputDir_HMM, 'arhmm_coef_AR1coef.csv'));
actmap = readmatrix(fullfile(rawDatDir, 'actSig_HCLindexed.csv'));

[numNeurons, numFrames] = size(hmm_binarymap);
numStates = max(hmm_statemap, [], 'all');
disp('Data loaded successfully.');

% --- Define the split point for time-axis plots ---
splitPoint = floor(numFrames / 2);
time_part1 = 1:splitPoint;
time_part2 = (splitPoint + 1):numFrames;


%% 2. POPULATION ANALYSIS: Activity Rates and Raster Plot (Split)
% -------------------------------------------------------------------------
disp('Generating population raster plots (in two parts)...');

% --- Part 1 of Raster Plot ---
figure('Position', [100 100 1200 600]);
imagesc(time_part1, 1:numNeurons, ~hmm_binarymap(:, time_part1));
colormap(gray);
title('Population Spike Raster Plot (Part 1)', 'FontSize', 16);
xlabel('Time Frame', 'FontSize', 12);
ylabel('Neuron ID', 'FontSize', 12);
yticks(1:20:numNeurons);
saveas(gcf, fullfile(analysisDir, 'population_raster_part1.png'));
saveas(gcf, fullfile(analysisDir, 'population_raster_part1.fig'));

% --- Part 2 of Raster Plot ---
figure('Position', [100 100 1200 600]);
imagesc(time_part2, 1:numNeurons, ~hmm_binarymap(:, time_part2));
colormap(gray);
title('Population Spike Raster Plot (Part 2)', 'FontSize', 16);
xlabel('Time Frame', 'FontSize', 12);
ylabel('Neuron ID', 'FontSize', 12);
yticks(1:20:numNeurons);
saveas(gcf, fullfile(analysisDir, 'population_raster_part2.png'));
saveas(gcf, fullfile(analysisDir, 'population_raster_part2.fig'));


% --- Activity Rate Histogram (this plot doesn't have a time axis) ---
disp('Calculating and plotting activity rates...');
activity_rate = sum(hmm_binarymap, 2) / numFrames;
figure('Position', [200 200 800 500]);
histogram(activity_rate * 100, 25);
title('Distribution of Neuron Activity Rates', 'FontSize', 16);
xlabel('Time spent in "Spike" state (%)', 'FontSize', 12);
ylabel('Number of Neurons', 'FontSize', 12);
grid on;
saveas(gcf, fullfile(analysisDir, 'activity_rate_histogram.png'));
saveas(gcf, fullfile(analysisDir, 'activity_rate_histogram.fig'));


%% 3. POPULATION ANALYSIS: Network Synchrony (Split)
% -------------------------------------------------------------------------
disp('Plotting population synchrony (in two parts)...');
population_synchrony = sum(hmm_binarymap, 1);

% --- Part 1 of Synchrony Plot ---
figure('Position', [300 300 1200 500]);
plot(time_part1, population_synchrony(time_part1), 'LineWidth', 1.5);
title('Population Synchrony Over Time (Part 1)', 'FontSize', 16);
xlabel('Time Frame', 'FontSize', 12);
ylabel('Number of Co-active Neurons', 'FontSize', 12);
xlim([time_part1(1), time_part1(end)]);
grid on;
saveas(gcf, fullfile(analysisDir, 'population_synchrony_part1.png'));
saveas(gcf, fullfile(analysisDir, 'population_synchrony_part1.fig'));

% --- Part 2 of Synchrony Plot ---
figure('Position', [300 300 1200 500]);
plot(time_part2, population_synchrony(time_part2), 'LineWidth', 1.5);
title('Population Synchrony Over Time (Part 2)', 'FontSize', 16);
xlabel('Time Frame', 'FontSize', 12);
ylabel('Number of Co-active Neurons', 'FontSize', 12);
xlim([time_part2(1), time_part2(end)]);
grid on;
saveas(gcf, fullfile(analysisDir, 'population_synchrony_part2.png'));
saveas(gcf, fullfile(analysisDir, 'population_synchrony_part2.fig'));


%% 4. STATE DYNAMICS: Characterizing the "Spike" State
% -------------------------------------------------------------------------
disp('Plotting AR(1) state parameters...');
spike_state_idx = size(intercepts, 2);
spike_intercepts = intercepts(:, spike_state_idx);
spike_ar1_coeffs = ar1_coeffs(:, spike_state_idx);

figure('Position', [400 400 700 600]);
scatter(spike_intercepts, spike_ar1_coeffs, 50, 'filled', 'MarkerFaceAlpha', 0.7);
title('AR(1) Parameters of the "Spike" State', 'FontSize', 16);
xlabel('Intercept (\phi_0) - Baseline drive', 'FontSize', 12);
ylabel('AR1 Coeff (\phi_1) - Persistence', 'FontSize', 12);
grid on;
refline(0,0);
xline(0);
saveas(gcf, fullfile(analysisDir, 'AR1_params_scatter.png'));
saveas(gcf, fullfile(analysisDir, 'AR1_params_scatter.fig'));


%% 5. DEEP DIVE: Visualizing a Single Neuron's Trace and States (Split)
% -------------------------------------------------------------------------
disp('Generating deep-dive plot for an example neuron (in two parts)...');
neuron_id_to_plot = 140;
raw_trace = actmap(neuron_id_to_plot, :);
state_sequence = hmm_statemap(neuron_id_to_plot, :);
colPal = jet(numStates);
legend_labels = [{'Raw Signal'}, arrayfun(@(s) ['State ' num2str(s)], 1:numStates, 'UniformOutput', false)];

% --- Part 1 of Single Neuron Plot ---
figure('Position', [100 500 1400 500]);
plot(time_part1, raw_trace(time_part1), 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
hold on;
for s = 1:numStates
    idx = find(state_sequence(time_part1) == s);
    scatter(time_part1(idx), raw_trace(time_part1(idx)), 30, colPal(s, :), 'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.5);
end
hold off;
title(['Neuron ', num2str(neuron_id_to_plot), ': Raw Trace and HMM States (Part 1)'], 'FontSize', 16);
xlabel('Time Frame', 'FontSize', 12);
ylabel('Calcium Activity', 'FontSize', 12);
xlim([time_part1(1), time_part1(end)]);
grid on;
legend(legend_labels, 'Location', 'northeastoutside');
filename_base_p1 = sprintf('neuron_%d_deep_dive_part1', neuron_id_to_plot);
saveas(gcf, fullfile(analysisDir, [filename_base_p1 '.png']));
saveas(gcf, fullfile(analysisDir, [filename_base_p1 '.fig']));

% --- Part 2 of Single Neuron Plot ---
figure('Position', [100 500 1400 500]);
plot(time_part2, raw_trace(time_part2), 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
hold on;
for s = 1:numStates
    idx = find(state_sequence(time_part2) == s);
    scatter(time_part2(idx), raw_trace(time_part2(idx)), 30, colPal(s, :), 'filled', 'MarkerEdgeColor','k', 'LineWidth', 0.5);
end
hold off;
title(['Neuron ', num2str(neuron_id_to_plot), ': Raw Trace and HMM States (Part 2)'], 'FontSize', 16);
xlabel('Time Frame', 'FontSize', 12);
ylabel('Calcium Activity', 'FontSize', 12);
xlim([time_part2(1), time_part2(end)]);
grid on;
legend(legend_labels, 'Location', 'northeastoutside');
filename_base_p2 = sprintf('neuron_%d_deep_dive_part2', neuron_id_to_plot);
saveas(gcf, fullfile(analysisDir, [filename_base_p2 '.png']));
saveas(gcf, fullfile(analysisDir, [filename_base_p2 '.fig']));


disp('Analysis complete.');
disp(['All plots saved to: ', analysisDir]);