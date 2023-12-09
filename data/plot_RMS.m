clear;
dly_m = importdata('RMS\delay_rms_1.6_model.txt')*1e6;
dly_d = importdata('RMS\delay_rms_1.6_data.txt')*1e6;

aoa_m = importdata('RMS\aoa_rms_1.6_model.txt');
aoa_d = importdata('RMS\aoa_rms_1.6_data.txt');

aod_m = importdata('RMS\aod_rms_1.6_model.txt');
aod_d = importdata('RMS\aod_rms_1.6_data.txt');

zoa_m = importdata('RMS\zoa_rms_1.6_model.txt');
zoa_d = importdata('RMS\zoa_rms_1.6_data.txt');

zod_m = importdata('RMS\zod_rms_1.6_model.txt');
zod_d = importdata('RMS\zod_rms_1.6_data.txt');

%t = tiledlayout(1,3);
%t.TileSpacing = 'compact';
figure();
%nexttile;
    h1 = plot(sort(dly_m), linspace(0,1,length(dly_m)));
    hold on;
    h2 = plot(sort(dly_d), linspace(0,1,length(dly_d)));
    hold on;
    grid on;
   % title('Delay RMS', 'FontSize',12)
    
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 13;
    ax.XAxis.FontSize = 13;
    set(gca,'fontname','times');
    set(gca,'fontweight','bold');
    ylabel('CDF', 'FontSize',15);
    
    set(h1, 'LineWidth',2.5);
    set(h1, 'Color', 'blue');
    set(h1, 'LineStyle',':');
    
    set(h2, 'LineWidth',2.0);
    set(h2, 'Color', 'red');
    set(h2, 'LineStyle','-');
    xticks([1:3:16]);
    yticks(linspace(0,1,11));
    xlabel('{\textbf{Delay RMS [$\mu s$]}}' ,'interpreter', 'latex','fontweight', 'bold','FontSize',15,'fontname', 'times');
    xlim([0,1])
    xticks(linspace(0,1.0, 6));
    
    %yticks([]);
    %h_ = legend([h1, h2], 'model', 'data', ...
    %'Location', 'west', 'fontweight','bold', ...
    %'fontname','times', ...
    %'fontsize', 12,...
    %'NumColumns', 2);

    %set(h_,'Position',[0.605333333333334 0.0265555550469293 0.244629843643423 0.0666666679382324]);
exportgraphics(gcf,'figures/rms_delay.png','Resolution',800);
%set(gcf,'Position',[100 100 900 300]);
%exportgraphics(gcf,'figures/rms_delay_aoa_aod.png','Resolution',800);    
figure();
%nexttile;
    h3 = plot(sort(aoa_m), linspace(0,1,length(aoa_m)));
    hold on;
    h4 = plot(sort(aoa_d), linspace(0,1,length(aoa_d)));
    hold on;
    grid on;
    %title('AOA RMS','FontSize',12)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 13;
    ax.XAxis.FontSize = 13;
    set(gca,'fontname','times');
    set(gca,'fontweight','bold');

    set(h3, 'LineWidth',2.5);
    set(h3, 'Color', 'blue');
    set(h3, 'LineStyle',':');
    
    set(h4, 'LineWidth',2.0);
    set(h4, 'Color', 'red');
    set(h4, 'LineStyle','-');
    set(gca,'yticklabel',[])
    % xticks([1:3:16]);
    %yticks(linspace(0,1,11));
    yticklabels([]);
    ylabel('');
     xlabel('{\textbf{AOA RMS [rad]}}' ,'interpreter', 'latex','fontweight', 'bold','FontSize',15,'fontname', 'times');
    exportgraphics(gcf,'figures/rms_aoa.png','Resolution',800);

%nexttile;
figure();
    h5 = plot(sort(aod_m), linspace(0,1,length(aod_m)));
    hold on;
    h6 = plot(sort(aod_d), linspace(0,1,length(aod_d)));
    hold on;
    grid on;
   % title('AOD RMS','FontSize',12)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 13;
    ax.XAxis.FontSize = 13;
    set(gca,'fontname','times')
    set(gca,'fontweight','bold')

    set(h5, 'LineWidth',2.5);
    set(h5, 'Color', 'blue');
    set(h5, 'LineStyle',':');

    set(h6, 'LineWidth',2.0);
    set(h6, 'Color', 'red');
    set(h6, 'LineStyle','-');
    set(gca,'yticklabel',[])
     %xticks([1:3:16]);
    %yticks(linspace(0,1,11));
    yticklabels([]);
    ylabel('');
    xlabel('{\textbf{AOD RMS [rad]}}' ,'interpreter', 'latex','fontweight', 'bold','FontSize',15,'fontname', 'times');
    exportgraphics(gcf,'figures/rms_aod.png','Resolution',800);


figure();
%nexttile;
    h7 = plot(sort(zoa_m), linspace(0,1,length(zoa_m)));
    hold on;
    h8 = plot(sort(zoa_d), linspace(0,1,length(zoa_d)));
    hold on;
    grid on;
    %title('AOA RMS','FontSize',12)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 13;
    ax.XAxis.FontSize = 13;
    set(gca,'fontname','times');
    set(gca,'fontweight','bold');

    set(h7, 'LineWidth',2.5);
    set(h7, 'Color', 'blue');
    set(h7, 'LineStyle',':');
    
    set(h8, 'LineWidth',2.0);
    set(h8, 'Color', 'red');
    set(h8, 'LineStyle','-');
    set(gca,'yticklabel',[])
    % xticks([1:3:16]);
    %yticks(linspace(0,1,11));
    yticklabels([]);
    ylabel('');
     xlabel('{\textbf{ZOA RMS [rad]}}' ,'interpreter', 'latex','fontweight', 'bold','FontSize',15,'fontname', 'times');
    exportgraphics(gcf,'figures/rms_zoa.png','Resolution',800);

%nexttile;
figure();
    h9 = plot(sort(zod_m), linspace(0,1,length(zod_m)));
    hold on;
    h10 = plot(sort(zod_d), linspace(0,1,length(zod_d)));
    hold on;
    grid on;
   % title('AOD RMS','FontSize',12)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 13;
    ax.XAxis.FontSize = 13;
    set(gca,'fontname','times')
    set(gca,'fontweight','bold')

    set(h9, 'LineWidth',2.5);
    set(h9, 'Color', 'blue');
    set(h9, 'LineStyle',':');

    set(h10, 'LineWidth',2.0);
    set(h10, 'Color', 'red');
    set(h10, 'LineStyle','-');
    set(gca,'yticklabel',[])
     %xticks([1:3:16]);
    %yticks(linspace(0,1,11));
    yticklabels([]);
    ylabel('');
    xlabel('{\textbf{ZOD RMS [rad]}}' ,'interpreter', 'latex','fontweight', 'bold','FontSize',15,'fontname', 'times');


    h_ = legend([h9, h10], 'model', 'data', ...
    'Location', 'southeast', 'fontweight','bold', ...
    'fontname','times', ...
    'fontsize', 18,...
    'NumColumns', 2);
    exportgraphics(gcf,'figures/rms_zod.png','Resolution',800);


%set(gcf,'Position',[100 100 900 300]);
%exportgraphics(gcf,'figures/rms_delay_aoa_aod.png','Resolution',800);
