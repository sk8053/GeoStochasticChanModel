clear;


heights = [1.6 30 60 90 120];
eps = true;
for j=1:length(heights)
    height = heights(j);
    if height == 1.6
        pl_m = importdata(sprintf('path_loss\\path_loss_%0.1f_model.txt',height));
        pl_d = importdata(sprintf('path_loss\\path_loss_%0.1f_data.txt',height));
    else
        pl_m = importdata(sprintf('path_loss\\path_loss_%d_model.txt',height));
        pl_d = importdata(sprintf('path_loss\\path_loss_%d_data.txt',height));
    end
    
    figure();
        h1 = plot(sort(pl_m), linspace(0,1,length(pl_m)));
        hold on;
        h2 = plot(sort(pl_d), linspace(0,1,length(pl_d)));
        hold on;
        grid on;
    
        if height == 1.6
            title(sprintf('height = %0.1f m',height), 'FontSize',18);
        else
            title(sprintf('height = %d m',height), 'FontSize',18);
        end
    
        xlim([80, 180])
        xticks(linspace(80, 180, 6));
    
        ax = gca;
        ax.GridLineWidth = 2;
        ax.YAxis.FontSize = 15;
        ax.XAxis.FontSize = 15;
        set(gca,'fontname','times');
        set(gca,'fontweight','bold');
        xlabel ('Pathloss [dB]', 'FontSize',17);
        ylabel('CDF', 'FontSize',17);
        
        set(h1, 'LineWidth',2.0);
        set(h1, 'Color', 'blue');
        set(h1, 'LineStyle',':');
        
        set(h2, 'LineWidth',2.0);
        set(h2, 'Color', 'red');
        set(h2, 'LineStyle','-');
        yticks(linspace(0,1,11));
        xline(160, 'k-', 'LineWidth',0.8);
        
        if height == 1.6
            save_file_name_png = 'figures/path_loss_1_6m.png';
            save_file_name_eps = 'figures/path_loss_1_6m.eps';
        else
            save_file_name_png = sprintf('figures//path_loss_%dm.png', height);
            save_file_name_eps = sprintf('figures//path_loss_%dm.eps', height);
        end

        if eps == true
            exportgraphics(gcf,save_file_name_eps);
        else
            exportgraphics(gcf,save_file_name_png,'Resolution', 800);
        end
end

