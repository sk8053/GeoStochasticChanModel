pl_1_6_m = importdata('path_loss\path_loss_1.6_model.txt');
pl_1_6_d = importdata('path_loss\path_loss_1.6_data.txt');

pl_30_m = importdata('path_loss\path_loss_30_model.txt');
pl_30_d = importdata('path_loss\path_loss_30_data.txt');

pl_60_m = importdata('path_loss\path_loss_60_model.txt');
pl_60_d = importdata('path_loss\path_loss_60_data.txt');

pl_90_m = importdata('path_loss\path_loss_90_model.txt');
pl_90_d = importdata('path_loss\path_loss_90_data.txt');

pl_120_m = importdata('path_loss\path_loss_120_model.txt');
pl_120_d = importdata('path_loss\path_loss_120_data.txt');


t = tiledlayout(1,5);
t.TileSpacing = 'compact';
%subplot(1,5,1);
nexttile;
    h1 = plot(sort(pl_1_6_m), linspace(0,1,length(pl_1_6_m)));
    hold on;
    h2 = plot(sort(pl_1_6_d), linspace(0,1,length(pl_1_6_d)));
    hold on;
    grid on;
    title('height = 1.6 m', 'FontSize',12)
    
    xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times');
    set(gca,'fontweight','bold')
    ylabel('CDF', 'FontSize',13);
    
    set(h1, 'LineWidth',2.0);
    set(h1, 'Color', 'blue');
    set(h1, 'LineStyle',':');
    
    set(h2, 'LineWidth',2.0);
    set(h2, 'Color', 'red');
    set(h2, 'LineStyle','-');
    yticks(linspace(0,1,11));
    h = legend([h1, h2], 'model', 'data', ...
    'Location', 'west', 'fontweight','bold','fontname', 'times', 'fontsize', 12, 'NumColumns', 2);   
    set(h, 'Position',[0.625333333333333 0.012988888030582 0.244629843643423 0.0666666679382324]);

nexttile;
    h3 = plot(sort(pl_30_m), linspace(0,1,length(pl_30_m)));
    hold on;
    h4 = plot(sort(pl_30_d), linspace(0,1,length(pl_30_d)));
    hold on;
    grid on;
    title('height = 30 m','FontSize',12)
    xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times')
    set(gca,'fontweight','bold')
    set(h3, 'LineWidth',2.0);
    set(h3, 'Color', 'blue');
    set(h3, 'LineStyle',':');
    set(h4, 'LineWidth',2.0);
    set(h4, 'Color', 'red');
    set(h4, 'LineStyle','-');
    set(gca,'yticklabel',[])
    
%subplot(1,5,3);
nexttile;
    h5 = plot(sort(pl_60_m), linspace(0,1,length(pl_60_m)));
    hold on;
    h6 = plot(sort(pl_60_d), linspace(0,1,length(pl_60_d)));
    hold on;
    grid on;
    title('height = 60 m','FontSize',12)
    xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times')
    set(gca,'fontweight','bold')
    set(h5, 'LineWidth',2.0);
    set(h5, 'Color', 'blue');
    set(h5, 'LineStyle',':');
    set(h6, 'LineWidth',2.0);
    set(h6, 'Color', 'red');
    set(h6, 'LineStyle','-');
    set(gca,'yticklabel',[])

%subplot(1,5,4);
nexttile;
    h7 = plot(sort(pl_90_m), linspace(0,1,length(pl_90_m)));
    hold on;
    h8 = plot(sort(pl_90_d), linspace(0,1,length(pl_90_d)));
    hold on;
    grid on;
    title('height = 90 m','FontSize',12)
    xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times')
    set(gca,'fontweight','bold')
    set(h7, 'LineWidth',2.0);
    set(h7, 'Color', 'blue');
    set(h7, 'LineStyle',':');
    set(h8, 'LineWidth',2.0);
    set(h8, 'Color', 'red');
    set(h8, 'LineStyle','-');
    set(gca,'yticklabel',[])
 
 %subplot(1,5,5);
 nexttile;
    h9 = plot(sort(pl_120_m), linspace(0,1,length(pl_120_m)));
    hold on;
    h10 = plot(sort(pl_120_d), linspace(0,1,length(pl_120_d)));
    hold on;
    grid on;
    title('height = 120 m','FontSize',12)
    xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    %ax.YAxis.FontSize = 10;
    %ax.XAxis.FontSize = 10;

    set(gca,'fontname','times')
    set(gca,'fontweight','bold')
    set(h9, 'LineWidth',2.0);
    set(h9, 'Color', 'blue');
    set(h9, 'LineStyle',':');
    set(h10, 'LineWidth',2.0);
    set(h10, 'Color', 'red');
    set(h10, 'LineStyle','-');
    set(gca,'yticklabel',[])

xlabel(t, 'Pathloss [dB]','FontWeight', 'bold','FontSize',13,'fontname', 'times');
set(gcf,'Position',[100 100 900 300]);
exportgraphics(gcf,'figures/path_loss.png','Resolution',800);
