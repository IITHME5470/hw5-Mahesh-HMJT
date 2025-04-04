% Configurations and folders
configs = {'serial', '2x2', '2x4', '4x4'};
folders = {'ser', '2_2', '2_4', '4_4'};

% Initialize table
timing_table = table();
timing_table.Config = configs';
timing_table.AvgTimePerStep = zeros(length(configs), 1);

% Read timing data
for c = 1:length(configs)
    folder = folders{c};
    timing_file = fullfile(folder, 'timing.txt');
    if ~exist(timing_file, 'file')
        error('Timing file not found: %s', timing_file);
    end
    
    fid = fopen(timing_file, 'r');
    line = fgetl(fid);
    fclose(fid);
    
    % Extract average time per step
    [~, num] = sscanf(line, '%*s Average time per time step = %e seconds');
    timing_table.AvgTimePerStep(c) = num;
end

% Display the table
disp('Time Taken Per Time Step for Serial and Parallel Runs:');
disp(timing_table);

% Save the table
writetable(timing_table, 'timing_per_step.txt', 'Delimiter', '\t');
disp('Table saved to timing_per_step.txt');