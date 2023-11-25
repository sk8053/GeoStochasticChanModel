clear;
az_1_6_m = importdata('azimuth_angles\aoa_1.6_model.txt');
az_1_6_d = importdata('azimuth_angles\aoa_1.6_data.txt');

az_30_m = importdata('azimuth_angles\aoa_30_model.txt');
az_30_d = importdata('azimuth_angles\aoa_30_data.txt');

az_60_m = importdata('azimuth_angles\aoa_60_model.txt');
az_60_d = importdata('azimuth_angles\aoa_60_data.txt');

az_90_m = importdata('azimuth_angles\aoa_90_model.txt');
az_90_d = importdata('azimuth_angles\aoa_90_data.txt');

az_120_m = importdata('azimuth_angles\aoa_120_model.txt');
az_120_d = importdata('azimuth_angles\aoa_120_data.txt');


t = tiledlayout(1,3);
t.TileSpacing = 'compact';
%subplot(1,5,1);
height = [1.6, 30, 60, 90, 120];
h_list = zeros(10,1);
colors = ['r','g','b','k','m'];
nexttile;
    for j =1:5
        if height(j) ==1.6
            data_d = importdata(sprintf('azimuth_angles/aoa_%.1f_data.txt',height(j)));
        else
            data_d = importdata(sprintf('azimuth_angles/aoa_%d_data.txt',height(j)));
        end
        legd = sprintf('model at %sm', num2str(height(j)));
        h2 = plot(sort(data_d), linspace(0,1,length(data_d)), 'DisplayName',legd);
        set(h2, 'LineWidth',2.0);
        set(h2, 'Color', colors(j));
        set(h2, 'LineStyle',':');
        hold on;
            
    end
    for j=1:5
        if height(j) ==1.6
            data_m = importdata(sprintf('azimuth_angles/aoa_%.1f_model.txt',height(j)));
        else
            data_m = importdata(sprintf('azimuth_angles/aoa_%d_model.txt',height(j)));
        end
        legd = sprintf('data at %sm', num2str(height(j)));
        h1 = plot(sort(data_m), linspace(0,1,length(data_m)), 'DisplayName',legd);
        set(h1, 'LineWidth',2.0);
        set(h1, 'Color', colors(j));
        set(h1, 'LineStyle','-');
        hold on;
        
    end
    ylabel ('CDF', 'FontSize',12);

    ax = gca;
    ax.GridLineWidth = 2;
    ax.FontSize = 10;
    set(gca,'fontname','times');
    title ('AOA [$^\circ$]', 'Interpreter','latex','FontSize',12);
    grid on;


nexttile;
     for j =1:5
        if height(j) ==1.6
            data_d = importdata(sprintf('azimuth_angles/aod_%.1f_data.txt',height(j)));
        else
            data_d = importdata(sprintf('azimuth_angles/aod_%d_data.txt',height(j)));
        end
        legd = sprintf('model at %sm', num2str(height(j)));
        h2 = plot(sort(data_d), linspace(0,1,length(data_d)), 'DisplayName',legd);
        set(h2, 'LineWidth',2.0);
        set(h2, 'Color', colors(j));
        set(h2, 'LineStyle',':');
        hold on;
            
    end
    for j=1:5
        if height(j) ==1.6
            data_m = importdata(sprintf('azimuth_angles/aod_%.1f_model.txt',height(j)));
        else
            data_m = importdata(sprintf('azimuth_angles/aod_%d_model.txt',height(j)));
        end
        legd = sprintf('data at %sm', num2str(height(j)));
        h1 = plot(sort(data_m), linspace(0,1,length(data_m)), 'DisplayName',legd);
        set(h1, 'LineWidth',2.0);
        set(h1, 'Color', colors(j));
        set(h1, 'LineStyle','-');
        hold on;
        
    end
 grid on;
title ('AOD [$^\circ$]', 'Interpreter','latex','FontSize',12);
ax = gca;
 ax.GridLineWidth = 2;
ax.FontSize = 10; 
 set(gca,'fontname','times');
set(gca,'yticklabel',[])

nexttile;
     for j =1:5
        if height(j) ==1.6
            data_d = importdata(sprintf('azimuth_angles/phase_%.1f_data.txt',height(j)));
        else
            data_d = importdata(sprintf('azimuth_angles/phase_%d_data.txt',height(j)));
        end
        legd = sprintf('model at %sm', num2str(height(j)));
        h2 = plot(sort(data_d), linspace(0,1,length(data_d)), 'DisplayName',legd);
        set(h2, 'LineWidth',2.0);
        set(h2, 'Color', colors(j));
        set(h2, 'LineStyle',':');
        hold on;
            
    end
    for j=1:5
        if height(j) ==1.6
            data_m = importdata(sprintf('azimuth_angles/phase_%.1f_model.txt',height(j)));
        else
            data_m = importdata(sprintf('azimuth_angles/phase_%d_model.txt',height(j)));
        end
        legd = sprintf('data at %sm', num2str(height(j)));
        h1 = plot(sort(data_m), linspace(0,1,length(data_m)), 'DisplayName',legd);
        set(h1, 'LineWidth',2.0);
        set(h1, 'Color', colors(j));
        set(h1, 'LineStyle','-');
        hold on;
        
    end
title ('Phase [$^\circ$]', 'Interpreter','latex','FontSize',12);
ax = gca;
ax.GridLineWidth = 2;
set(gca,'fontname','times');
set(gca,'yticklabel',[])

xlabel(t, 'Angle [$^\circ$]','FontSize',12,'fontname', 'times','Interpreter','Latex');
ax = gca;
ax.FontSize = 10; 
set(gcf,'Position',[100 100 800 300]);
grid on;
lgd = legend ('NumColumns',1,'Location','southeast','Fontsize', 10, 'Location','eastoutside','fontname','times');
%set(lgd,...
%    'Position',[0.913916664044062 0.293555555809869 0.173750002622604 0.588333346684774],...
%    'FontSize',11);
legend show;
exportgraphics(gcf,'figures/azimuth_phase.png','Resolution',800);
   