dly_1_6_m = importdata('delay\delay_1.6_model.txt')*1e6;
dly_1_6_d = importdata('delay\delay_1.6_data.txt')*1e6;

dly_30_m = importdata('delay\delay_30_model.txt')*1e6;
dly_30_d = importdata('delay\delay_30_data.txt')*1e6;

dly_60_m = importdata('delay\delay_60_model.txt')*1e6;
dly_60_d = importdata('delay\delay_60_data.txt')*1e6;

dly_90_m = importdata('delay\delay_90_model.txt')*1e6;
dly_90_d = importdata('delay\delay_90_data.txt')*1e6;

dly_120_m = importdata('delay\delay_120_model.txt')*1e6;
dly_120_d = importdata('delay\delay_120_data.txt')*1e6;


t = tiledlayout(1,5);
t.TileSpacing = 'compact';

nexttile;
    h1 = plot(sort(dly_1_6_m), linspace(0,1,length(dly_1_6_m)));
    hold on;
    h2 = plot(sort(dly_1_6_d), linspace(0,1,length(dly_1_6_d)));
    hold on;
    grid on;
    title('height = 1.6 m', 'FontSize',12)
    
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 11;
    ax.XAxis.FontSize = 11;
    set(gca,'fontname','times');
    set(gca,'fontweight','bold');
    ylabel('CDF', 'FontSize',13);
    
    set(h1, 'LineWidth',2.5);
    set(h1, 'Color', 'blue');
    set(h1, 'LineStyle',':');
    
    set(h2, 'LineWidth',2.0);
    set(h2, 'Color', 'red');
    set(h2, 'LineStyle','-');
    xticks([1:3:16]);
    yticks(linspace(0,1,11));
    h_ = legend([h1, h2], 'model', 'data', ...
    'Location', 'west', 'fontweight','bold', ...
    'fontname','times', ...
    'fontsize', 12,...
    'NumColumns', 2);

    set(h_,'Position',[0.605333333333334 0.0265555550469293 0.244629843643423 0.0666666679382324]);
    

nexttile;
    h3 = plot(sort(dly_30_m), linspace(0,1,length(dly_30_m)));
    hold on;
    h4 = plot(sort(dly_30_d), linspace(0,1,length(dly_30_d)));
    hold on;
    grid on;
    title('height = 30 m','FontSize',12)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 11;
    ax.XAxis.FontSize = 11;
    set(gca,'fontname','times');
    set(gca,'fontweight','bold');

    set(h3, 'LineWidth',2.5);
    set(h3, 'Color', 'blue');
    set(h3, 'LineStyle',':');
    
    set(h4, 'LineWidth',2.0);
    set(h4, 'Color', 'red');
    set(h4, 'LineStyle','-');
    set(gca,'yticklabel',[])
     xticks([1:3:16]);
    yticks(linspace(0,1,11));

nexttile;
    h5 = plot(sort(dly_60_m), linspace(0,1,length(dly_60_m)));
    hold on;
    h6 = plot(sort(dly_60_d), linspace(0,1,length(dly_60_d)));
    hold on;
    grid on;
    title('height = 60 m','FontSize',12)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 11;
    ax.XAxis.FontSize = 11;
    set(gca,'fontname','times')
    set(gca,'fontweight','bold')

    set(h5, 'LineWidth',2.5);
    set(h5, 'Color', 'blue');
    set(h5, 'LineStyle',':');

    set(h6, 'LineWidth',2.0);
    set(h6, 'Color', 'red');
    set(h6, 'LineStyle','-');
    set(gca,'yticklabel',[])
     xticks([1:3:16]);
    yticks(linspace(0,1,11));

nexttile;
    h7 = plot(sort(dly_90_m), linspace(0,1,length(dly_90_m)));
    hold on;
    h8 = plot(sort(dly_90_d), linspace(0,1,length(dly_90_d)));
    hold on;
    grid on;
    title('height = 90 m','FontSize',12)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 11;
    ax.XAxis.FontSize = 11;
    set(gca,'fontname','times');
    set(gca,'fontweight','bold');

    set(h7, 'LineWidth',2.5);
    set(h7, 'Color', 'blue');
    set(h7, 'LineStyle',':');

    set(h8, 'LineWidth',2.0);
    set(h8, 'Color', 'red');
    set(h8, 'LineStyle','-');
    set(gca,'yticklabel',[])
    xticks([1:3:16]);
    yticks(linspace(0,1,11));

 nexttile;
    h9 = plot(sort(dly_120_m), linspace(0,1,length(dly_120_m)));
    hold on;
    h10 = plot(sort(dly_120_d), linspace(0,1,length(dly_120_d)));
    hold on;
    grid on;
    title('height = 120 m','FontSize',12)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 11;
    ax.XAxis.FontSize = 11;
    set(gca,'fontname','times')
    set(gca,'fontweight','bold');
    set(h9, 'LineWidth',2.5);
    set(h9, 'Color', 'blue');
    set(h9, 'LineStyle',':');
    
    set(h10, 'LineWidth',2.0);
    set(h10, 'Color', 'red');
    set(h10, 'LineStyle','-');
    set(gca,'yticklabel',[])
     xticks([1:3:16]);
    yticks(linspace(0,1,11));

    %set(h1, 'Position', [0,1,2,3,1,1,1,1])
xlabel(t, '{\textbf{Delay [$\mu s$]}}' ,'interpreter', 'latex','fontweight', 'bold','FontSize',13,'fontname', 'times');
set(gcf,'Position',[100 100 900 300]);
exportgraphics(gcf,'figures/delay.png','Resolution',800);
