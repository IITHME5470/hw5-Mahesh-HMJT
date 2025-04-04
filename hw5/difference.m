% Configurations and folders
configs = {'serial', '2x2', '2x4', '4x4'};
proc_grids = {[1, 1], [2, 2], [2, 4], [4, 4]};
folders = {'ser', '2_2', '2_4', '4_4'};
tid = 5107; % 10th time step from your list

% Load serial data
serial_file = fullfile('ser', sprintf('T_x_y_%06d.dat', tid));
if ~exist(serial_file, 'file')
    error('Serial file not found: %s', serial_file);
end
data_serial = dlmread(serial_file);
n = sqrt(size(data_serial, 1)); % 800x800 grid
T_serial = reshape(data_serial(:, 3), [n, n]);

% Initialize table data
diff_table = table();
diff_table.Config = configs(2:end)'; % Exclude serial from table rows
diff_table.MaxDiff = zeros(length(configs)-1, 1);
diff_table.MeanDiff = zeros(length(configs)-1, 1);
diff_table.RMSDiff = zeros(length(configs)-1, 1);

% Process each parallel configuration
for c = 2:length(configs) % Start from 2 to skip serial
    config = configs{c};
    px = proc_grids{c}(1);
    py = proc_grids{c}(2);
    folder = folders{c};

    all_x = [];
    all_y = [];
    all_T = [];

    % Collect data from all ranks
    for px_idx = 0:px-1
        for py_idx = 0:py-1
            rank = px_idx * py + py_idx;
            filename = fullfile(folder, sprintf('T_x_y_%06d_%02d.dat', tid, rank));
            if exist(filename, 'file')
                data = dlmread(filename);
                all_x = [all_x; data(:, 1)];
                all_y = [all_y; data(:, 2)];
                all_T = [all_T; data(:, 3)];
            else
                warning(['File not found: ', filename]);
            end
        end
    end

    if isempty(all_x)
        error('No data for %s at t = %d', config, tid);
    end

    % Interpolate onto serial grid
    unique_x = unique(all_x);
    unique_y = unique(all_y);
    [X, Y] = meshgrid(unique_x, unique_y);
    F = scatteredInterpolant(all_x, all_y, all_T, 'natural');
    T_parallel = F(X, Y);

    % Ensure T_parallel matches T_serial size (800x800)
    if size(T_parallel) ~= size(T_serial)
        error('Grid size mismatch for %s: expected %dx%d, got %dx%d', ...
              config, n, n, size(T_parallel, 1), size(T_parallel, 2));
    end

    % Compute differences
    diff = abs(T_serial - T_parallel);
    diff_table.MaxDiff(c-1) = max(diff(:));
    diff_table.MeanDiff(c-1) = mean(diff(:));
    diff_table.RMSDiff(c-1) = sqrt(mean(diff(:).^2));
end

% Display the table
disp('Differences between Serial and Parallel Runs at Time Step 5107:');
disp(diff_table);

% Save the table to a file
writetable(diff_table, 'differences_at_5107.txt', 'Delimiter', '\t');
disp('Table saved to differences_at_5107.txt');