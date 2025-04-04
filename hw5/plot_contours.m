
tid = 5107;
plot_serial_results(tid)

function plot_serial_results(tid)
    % Read data
    filename = sprintf('T_x_y_%06d.dat', tid);
    a = dlmread(filename);
    
    % Reshape data
    n = sqrt(size(a,1));
    if mod(n,1) ~= 0
        error('Data size is not perfect square');
    end
    
    x = a(1:n:n^2,1);
    y = a(1:n,2);
    T = reshape(a(:,3), [n, n]);
    
    % Verify grid
    if ~issorted(x) || ~issorted(y)
        error('Grid coordinates are not monotonically increasing');
    end
    
    % Plot
    figure, clf
    contourf(x,y,T','LineColor', 'none')
    xlabel('x'), ylabel('y'), title(sprintf('t = %06d', tid));
    xlim([min(x)-0.05 max(x)+0.05])
    ylim([min(y)-0.05 max(y)+0.05])
    caxis([-0.05 1.05]), colorbar
    colormap('jet')
    set(gca, 'FontSize', 14)
end
