clear;

heights = [1.6 30 60 90 120];
eps = false;

for j=1:length(heights)
    
    height = heights(j);

    if height ==1.6
        ls_m = importdata(sprintf('link state\\link state_%.1f_model.txt', height));
        ls_d = importdata(sprintf('link state\\link state_%.1f_data.txt', height));
        ot_m = importdata(sprintf('link state\\outage state_%.1f_model.txt', height));
        ot_d = importdata(sprintf('link state\\outage state_%.1f_data.txt', height));
        
    else
        ls_m = importdata(sprintf('link state\\link state_%d_model.txt', height));
        ls_d = importdata(sprintf('link state\\link state_%d_data.txt', height));
        ot_m = importdata(sprintf('link state\\outage state_%d_model.txt', height));
        ot_d = importdata(sprintf('link state\\outage state_%d_data.txt', height));
    
    end
    
    figure(); % 1.6m
    
        h1 = plot((1:length(ls_m))*10, ls_m);
        hold on;
        h2 = plot((1:length(ls_d))*10, ls_d);
        hold on;
    
        h1_ot = plot((1:length(ot_m))*10, ot_m);
        hold on;
        h2_ot = plot((1:length(ot_d))*10, ot_d);
        hold on;
     
    
        grid on;
    
        if height == 1.6
            title(sprintf('height = %0.1f m',height), 'FontSize',18);
        else
            title(sprintf('height = %d m',height), 'FontSize',18);
        end
    
        ax = gca;
        ax.GridLineWidth = 2;
        ax.YAxis.FontSize = 15;
        ax.XAxis.FontSize = 15;
    
        set(gca,'fontname','times');
        set(gca,'fontweight','bold');
        if height ==1.6
            ylabel('Proability', 'FontSize',17);
        end
        
        set(h1, 'LineWidth',2.0);
        set(h1, 'Color', 'blue');
        set(h1, 'LineStyle',':');
        
        set(h2, 'LineWidth',2.0);
        set(h2, 'Color', 'red');
        set(h2, 'LineStyle','-');
    
        set(h1_ot, 'LineWidth',2.0);
        
        h1_ot.Color =[0.0745    0.6235    1.0000];
        set(h1_ot, 'LineStyle',':');
        
        set(h2_ot, 'LineWidth',2.0);
        h2_ot.Color = [0.9804    0.3922    0.3922];
        set(h2_ot, 'LineStyle','-');
         
        xlim([0, 1000]);
        xticks([0:250:1200]);
        yticks(linspace(0,1,11));
        
        xlabel('2D distance [m]','fontweight', 'bold','FontSize',17,'fontname', 'times');
    
        if height ==60
            x = [0.3 0.195];
            y = [0.75 0.7];
            annotation('textarrow',x,y,'String',' LOS','FontSize',17,'FontWeight','bold','Linewidth',3)
            x = [0.7 0.815];
            y = [0.75 0.66];
            annotation('textarrow',x,y,'String','Outage','FontSize',17,'FontWeight','bold','Linewidth',3);
        end
    
         if height == 1.6
            save_file_name_png = 'figures/path_loss_1_6m.png';
            save_file_name_eps = 'figures/path_loss_1_6m.eps';
        else
            save_file_name_png = sprintf('figures//path_loss_%dm.png', height);
            save_file_name_eps = sprintf('figures//path_loss_%dm.eps', height);
        end
        
        if height == 120
                h = legend([h1, h2], 'model', 'data', ...
            'Location', 'northeast','fontweight','bold', 'fontsize', 15,  'box','on','NumColumns', 1);
        end
    
         if eps == true
            exportgraphics(gcf,save_file_name_eps);
        else
            exportgraphics(gcf,save_file_name_png,'Resolution', 800);
        end
end
