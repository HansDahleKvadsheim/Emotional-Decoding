% Window-slide method of dynamic functional connectivity for CSV files
% Input: ROI signals in CSV format
% Output: Dynamic functional connectivity matrices

%% Settings
FZTInd = 1; % if output in z-score
SWS = 12;    % Window length

%timestamps = [26.43647366, 33.29379116, 38.3141819, 46.32470944, 57.14761392, 69.18570055, 77.13153959, 85.67322187, 93.46303143, 105.4122276, 113.024486, 118.9260682, 126.5552283, 137.9927443, 144.8481177, 151.07709, 159.8622162, 165.1467442, 172.3685173, 180.5430883, 189.9755977, 201.4791154, 208.5528044, 221.9107697, 228.161039, 235.9587606, 242.3571085, 250.0138598, 255.6883308, 264.0836428, 271.060261, 280.2411218, 287.2736964, 293.6633895, 301.0755974, 310.1689731, 317.1250291, 328.6828049, 335.3501167, 341.8209622, 350.088132, 358.4149621, 364.3036738, 372.1732749, 380.5302472, 386.8517698, 394.6300104, 401.1438242, 411.1592425, 416.0115255, 423.4433681, 433.6796761, 439.1828763, 446.9767341, 453.1260618, 464.3987982, 471.7981782, 478.3289987, 487.4939448, 495.8518498, 503.7178932, 515.3396564, 521.6594818, 527.0084163, 535.3188653, 539.9397492, 549.3023919, 557.3862353, 565.5371617, 578.0013664, 584.8966744, 595.9690811, 602.0143757, 610.9729594, 622.3398328, 628.6588845, 635.1190617, 640.2352063, 649.105529, 654.9406018, 664.8399321, 675.4552575, 687.0087004, 696.3920602, 702.1155328, 709.3927927, 719.734513, 723.8392682, 731.3270649];

timestamps_path = 'data/timestamps/expanded_timestamps.csv';

if isfile(timestamps_path)
    table = readtable(timestamps_path);
else
    error('File "%s" does not exist. Please check the path.', timestamps_path);
end

timestamps = table.timestamps;

% Define paths
% data_path = 'data/PRUNED_ROI_signals';
data_path = 'data/ROI_signals/ROIs';
save_path = 'Alice/Results/DFC/DFCMatrixZ';

if ~isfolder(save_path)
    mkdir(save_path);
end

% Get list of CSV files
csv_files = dir(fullfile(data_path, '*.csv'));

disp(['Number of files: ', num2str(length(csv_files))]);


for i = 1:length(csv_files)
    % Load ROI signal from CSV file
    csv_file = fullfile(data_path, csv_files(i).name);
    TC = readmatrix(csv_file);  % read CSV into matrix
    TC = TC(:, 2:end);          % keep only columns 2 through the end
 

    disp(['Processing file: ', csv_files(i).name]);

    TP = size(TC, 1); % Number of time points
    DRStruct = [];
    if FZTInd == 1
        DZStruct = [];
    end

    for s = 1:length(timestamps)
        element = timestamps(s);
        First = fix(element / 2);
        Last = First + SWS;
        % disp(["Last: ", Last])
        % disp(["TP: ", TP])
        if Last > TP
            disp('Window exceeds signal length, skipping...');
            continue;
        else
            disp(['Processing timestamp: ', num2str(element)]);
        end
        
        WinInd = (First:Last)';
        Tag = sprintf('W_%.4d_%.4d', First, Last);
        DTC = TC(WinInd, :);
        DR = corr(DTC, 'type', 'Pearson');
        DR = (DR + DR') / 2; % Symmetrize
        DR(isnan(DR)) = 0;
        DRStruct.(Tag) = DR;

        if FZTInd == 1
            DR(DR >= 1) = 1 - 1e-16; % Avoid Inf in z-score calculation
            DZ = 0.5 * log((1 + DR) ./ (1 - DR));
            DZStruct.(Tag) = DZ;
        end
        R_Tag{s} = Tag;
        
    end
    % Save results
    DRFCFile = fullfile(save_path, ['r_', csv_files(i).name, '.mat']);
    save(DRFCFile, 'DRStruct');

    if FZTInd == 1
        DZFCFile = fullfile(save_path, ['z_', csv_files(i).name, '.mat']);
        save(DZFCFile, 'DZStruct');
    end
end
