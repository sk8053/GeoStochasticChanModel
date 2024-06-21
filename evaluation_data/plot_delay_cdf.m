clear;

heights = [1.6 30 60 90 120];
eps = false;

for j=1:length(heights)
    height = heights(j);
    if height == 1.6
        dly_m = importdata(sprintf('delay\\delay_%0.1f_model.txt',height))*1e6;
        dly_d = importdata(sprintf('delay\\delay_%0.1f_data.txt',height))*1e6;
    else
        dly_m = importdata(sprintf('delay\\delay_%d_model.txt',height))*1e6;
        dly_d = importdata(sprintf('delay\\delay_%d_data.txt',height))*1e6;
    end
       
    figure();
        h1 = plot(sort(dly_m), linspace(0,1,length(dly_m)));
        hold on;
        h2 = plot(sort(dly_d), linspace(0,1,length(dly_d)));
        hold on;
        grid on;
        
        if height == 1.6
             title(sprintf('height = %0.1f m',height), 'FontSize',18);
        else
             title(sprintf('height = %d m',height), 'FontSize',18);
        end
        
        
        %xticks(linspace(80, 200, 7));
        ax = gca;
        ax.GridLineWidth = 2;
        ax.YAxis.FontSize = 15;
        ax.XAxis.FontSize = 15;
    
        set(gca,'fontname','times');
        set(gca,'fontweight','bold');
        ylabel('CDF', 'FontSize',17);
        
        set(h1, 'LineWidth',2.0);
        set(h1, 'Color', 'blue');
        set(h1, 'LineStyle',':');
        
        set(h2, 'LineWidth',2.0);
        set(h2, 'Color', 'red');
        set(h2, 'LineStyle','-');
        xlim([0,11]);
        xticks([1:3:16]);
        yticks(linspace(0,1,11));
    
        xline(1, 'k-', 'LineWidth',0.8);
        %set(h_,'Position',[0.605333333333334 0.0265555550469293 0.244629843643423 0.0666666679382324]);
        xlabel('{\textbf{Delay [$\mu s$]}}' ,'interpreter', 'latex','fontweight', 'bold','FontSize',17,'fontname', 'times');
        
        if height == 1.6
            save_file_name_png = 'figures/delay_1_6m.png';
            save_file_name_eps = 'figures/delay_1_6m.eps';
    
        else
            save_file_name_png = sprintf('figures//delay_%dm.png', height);
            save_file_name_eps = sprintf('figures//delay_%dm.eps', height);
        end
    
        if eps == true
            exportgraphics(gcf,save_file_name_eps);
        else
            exportgraphics(gcf,save_file_name_png,'Resolution', 800);
        end
end
