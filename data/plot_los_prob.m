ls_1_6_m = importdata('link state\link state_1.6_model.txt');
ls_1_6_d = importdata('link state\link state_1.6_data.txt');

ls_30_m = importdata('link state\link state_30_model.txt');
ls_30_d = importdata('link state\link state_30_data.txt');

ls_60_m = importdata('link state\link state_60_model.txt');
ls_60_d = importdata('link state\link state_60_data.txt');

ls_90_m = importdata('link state\link state_90_model.txt');
ls_90_d = importdata('link state\link state_90_data.txt');

ls_120_m = importdata('link state\link state_120_model.txt');
ls_120_d = importdata('link state\link state_120_data.txt');


t = tiledlayout(1,5);
t.TileSpacing = 'compact';
%subplot(1,5,1);
nexttile;
%np.arange(len(los_prob_model)*10, step = 10), los_prob_model
    h1 = plot((1:length(ls_1_6_m))*10, ls_1_6_m);
    hold on;
    h2 = plot((1:length(ls_1_6_d))*10, ls_1_6_d);
    hold on;
    grid on;
    title('height = 1.6 m', 'FontSize',10)
    
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times');

    ylabel('CDF', 'FontSize',11);
    
    set(h1, 'LineWidth',2.0);
    set(h1, 'Color', 'blue');
    set(h1, 'LineStyle',':');
    
    set(h2, 'LineWidth',2.0);
    set(h2, 'Color', 'red');
    set(h2, 'LineStyle','-');
    
    h = legend([h1, h2], 'model', 'data', ...
    'Location', 'west', 'interpreter', 'latex', 'fontsize', 10,  'box','on','NumColumns', 2);
    %legend1 = legend(axes1,'show');
    set(h,...
    'Position',[0.625333333333333 0.011888888030582 0.244629843643423 0.0666666679382324],...
    'NumColumns',2,...
    'Interpreter','latex',...
    'FontSize',10);
    xlim([0, 1200])
    %legend([h1, h2], 'model', 'data', ...
    %    'Location', 'northwest', 'interpreter', 'latex', 'fontsize', 10);
%subplot(1,5,2);
nexttile;
    h3 = plot((1:length(ls_30_m))*10, ls_30_m);
    hold on;
    h4 = plot((1:length(ls_30_d))*10, ls_30_d);
    hold on;
    grid on;
    title('height = 30 m','FontSize',10)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times')

    set(h3, 'LineWidth',2.0);
    set(h3, 'Color', 'blue');
    set(h3, 'LineStyle',':');
    set(h4, 'LineWidth',2.0);
    set(h4, 'Color', 'red');
    set(h4, 'LineStyle','-');
    set(gca,'yticklabel',[])
    xlim([0, 1200])
%subplot(1,5,3);
nexttile;
    h5 = plot((1:length(ls_60_m))*10, ls_60_m);
    hold on;
    h6 = plot((1:length(ls_60_d))*10, ls_60_d);
    hold on;
    grid on;
    title('height = 60 m','FontSize',10)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times')
    
    set(h5, 'LineWidth',2.0);
    set(h5, 'Color', 'blue');
    set(h5, 'LineStyle',':');
    set(h6, 'LineWidth',2.0);
    set(h6, 'Color', 'red');
    set(h6, 'LineStyle','-');
    set(gca,'yticklabel',[])
    xlim([0, 1200])
%subplot(1,5,4);
nexttile;
    h7 = plot((1:length(ls_90_m))*10, ls_90_m);
    hold on;
    h8 = plot((1:length(ls_90_d))*10, ls_90_d);
    hold on;
    grid on;
    title('height = 90 m','FontSize',10)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times')

    set(h7, 'LineWidth',2.0);
    set(h7, 'Color', 'blue');
    set(h7, 'LineStyle',':');
    set(h8, 'LineWidth',2.0);
    set(h8, 'Color', 'red');
    set(h8, 'LineStyle','-');
    set(gca,'yticklabel',[])
    xlim([0, 1200])
 %subplot(1,5,5);
 nexttile;
    h9 = plot((1:length(ls_120_m))*10, ls_120_m);
    hold on;
    h10 =plot((1:length(ls_120_d))*10, ls_120_d);
    hold on;
    grid on;
    title('height = 120 m','FontSize',10)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    set(gca,'fontname','times')

    set(h9, 'LineWidth',2.0);
    set(h9, 'Color', 'blue');
    set(h9, 'LineStyle',':');
    set(h10, 'LineWidth',2.0);
    set(h10, 'Color', 'red');
    set(h10, 'LineStyle','-');
    set(gca,'yticklabel',[])
    xlim([0, 1200])
xlabel(t, '2D distance [m]','FontSize',12,'fontname', 'times');
set(gcf,'Position',[100 100 800 300]);
exportgraphics(gcf,'figures/los_prob.png','Resolution',800);
