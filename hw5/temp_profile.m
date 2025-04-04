% Define time steps and configurations
time_steps = [1020, 3060, 5107]; % Example time steps (adjust based on your outputs)
configs = {'serial', '2x2', '2x4', '4x4'};
proc_grids = {[1, 1], [2, 2], [2, 4], [4, 4]}; % [px, py] for each config
folders = {'ser', '2_2', '2_4', '4_4'}; % Corresponding folder names

% Loop over each time step
for tid = time_steps
    figure('Position', [100, 100, 800, 600]); % New figure for each time step
    hold on;

    % Process each configuration
    for c = 1:length(configs)
        config = configs{c};
        px = proc_grids{c}(1);
        py = proc_grids{c}(2);
        folder = folders{c}; % Get the folder name for this config

        if strcmp(config, 'serial')
            % Serial case
            filename = fullfile(folder, sprintf('T_x_y_%06d.dat', tid));
            if ~exist(filename, 'file')
                warning(['Serial file not found: ', filename]);
                continue;
            end
            data = dlmread(filename);
            n = sqrt(size(data, 1)); % Assuming square grid (800x800)
            x = data(1:n:end, 1);   % x-coordinates
            T = reshape(data(:, 3), [n, n]);
            mid_idx = round(n / 2);  % Mid-plane at y = 0.5
            T_mid = T(mid_idx, :);  % Temperature along y = 0.5

            plot(x, T_mid, '-', 'LineWidth', 2, 'DisplayName', 'Serial');
        else
            % Parallel case
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
                warning(['No data for ', config, ' at t = ', num2str(tid)]);
                continue;
            end

            % Get unique x-coordinates and interpolate along y = 0.5
            unique_x = unique(all_x);
            mid_y = 0.5 * ones(size(unique_x)); % y = 0.5 for all x
            F = scatteredInterpolant(all_x, all_y, all_T, 'natural');
            T_mid = F(unique_x, mid_y); % Interpolate T at (x, y = 0.5)

            % Plot with dashed line for parallel cases
            plot(unique_x, T_mid, '--', 'LineWidth', 2, 'DisplayName', config);
        end
    end

    % Format the plot
    xlabel('x');
    ylabel('Temperature (T)');
    title(sprintf('Temperature Profile at y = 0.5, t = %.6f', tid * 1.96e-7)); % Assuming dt ≈ 1.96e-7
    xlim([0 1]);
    ylim([0 1.2]); % Adjust based on expected range (initial max is 0.25)
    legend('show', 'Location', 'best');
    grid on;
    set(gca, 'FontSize', 14);
    hold off;

    % Save the plot
    saveas(gcf, sprintf('profile_comparison_%06d.png', tid));
    disp(['Saved profile plot for t = ', num2str(tid)]);
end