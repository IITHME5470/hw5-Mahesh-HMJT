% Call the function with the correct time step and processor grid
plot_combined_results(5107, 2, 2); 

function plot_combined_results(tid, num_procs_x, num_procs_y)
    % Initialize combined arrays
    all_x = [];
    all_y = [];
    all_T = [];
    
    % Collect data from all ranks
    for px = 0:num_procs_x-1
        for py = 0:num_procs_y-1
            rank = px * num_procs_y + py; % Rank calculation for 2x2 grid: 0, 1, 2, 3
            filename = sprintf('T_x_y_%06d_%02d.dat', tid, rank);
            
            if exist(filename, 'file')
                disp(['Reading: ', filename]);
                data = dlmread(filename);
                
                all_x = [all_x; data(:,1)];
                all_y = [all_y; data(:,2)];
                all_T = [all_T; data(:,3)];
            else
                warning(['File not found: ', filename]);
            end
        end
    end
    
    % Check if data was loaded
    if isempty(all_x)
        error('No data files found for time step %d', tid);
    end
    
    % Get unique coordinates (sorted)
    unique_x = unique(all_x);
    unique_y = unique(all_y);
    
    % Create grid and interpolate
    [X, Y] = meshgrid(unique_x, unique_y);
    F = scatteredInterpolant(all_x, all_y, all_T, 'natural');
    T_combined = F(X, Y);
    
    % Plot contour
    figure; % Larger figure size for clarity
    clf;
    contourf(X, Y, T_combined, 20, 'LineColor', 'none'); % 20 levels for smoother contours
    xlabel('x');
    ylabel('y');
    title(sprintf('Temperature Distribution at t = %.6f', tid * 1.96e-7)); % Assuming dt ≈ 1.96e-7
    xlim([min(unique_x) max(unique_x)]);
    ylim([min(unique_y) max(unique_y)]);
    caxis([-0.05 1.05]); 
    colorbar;
    colormap('jet');
    set(gca, 'FontSize', 14);
    
    % Save the plot
    saveas(gcf, sprintf('combined_plot_%06d.png', tid));
    
    % % Save combined data in same format as serial version
    % fid = fopen(sprintf('T_x_y_%06d_combined.dat', tid), 'w');
    % for i = 1:length(unique_x)
    %     for j = 1:length(unique_y)
    %         fprintf(fid, '%f %f %f\n', unique_x(i), unique_y(j), T_combined(j,i));
    %     end
    % end
    % fclose(fid);
    % disp(['Combined data saved to T_x_y_%06d_combined.dat', tid]);
end