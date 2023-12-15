clear;
ls_1_6_m = importdata('link state\link state_1.6_model.txt');
ls_1_6_d = importdata('link state\link state_1.6_data.txt');

ot_1_6_m = importdata('link state\outage state_1.6_model.txt');
ot_1_6_d = importdata('link state\outage state_1.6_data.txt');

ls_30_m = importdata('link state\link state_30_model.txt');
ls_30_d = importdata('link state\link state_30_data.txt');

ot_30_m = importdata('link state\outage state_30_model.txt');
ot_30_d = importdata('link state\outage state_30_data.txt');


ls_60_m = importdata('link state\link state_60_model.txt');
ls_60_d = importdata('link state\link state_60_data.txt');

ot_60_m = importdata('link state\outage state_60_model.txt');
ot_60_d = importdata('link state\outage state_60_data.txt');

ls_90_m = importdata('link state\link state_90_model.txt');
ls_90_d = importdata('link state\link state_90_data.txt');

ot_90_m = importdata('link state\outage state_90_model.txt');
ot_90_d = importdata('link state\outage state_90_data.txt');

ls_120_m = importdata('link state\link state_120_model.txt');
ls_120_d = importdata('link state\link state_120_data.txt');

ot_120_m = importdata('link state\outage state_120_model.txt');
ot_120_d = importdata('link state\outage state_120_data.txt');

eps = false;
%t = tiledlayout(1,5);
%t.TileSpacing = 'compact';
%subplot(1,5,1);
%nexttile;
figure();
%np.arange(len(los_prob_model)*10, step = 10), los_prob_model
    h1 = plot((1:length(ls_1_6_m))*10, ls_1_6_m);
    hold on;
    h2 = plot((1:length(ls_1_6_d))*10, ls_1_6_d);
    hold on;

    h1_ot = plot((1:length(ot_1_6_m))*10, ot_1_6_m);
    hold on;
    h2_ot = plot((1:length(ot_1_6_d))*10, ot_1_6_d);
    hold on;
    %h2_ot_m = plot((1:20:length(ot_1_6_d))*10, ot_1_6_d);

    grid on;
    title('height = 1.6 m', 'FontSize',18, 'FontWeight','bold')
    
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 15;
    ax.XAxis.FontSize = 15;

    set(gca,'fontname','times');
    set(gca,'fontweight','bold');
    ylabel('Proability', 'FontSize',17);
    
    set(h1, 'LineWidth',2.0);
    set(h1, 'Color', 'blue');
    set(h1, 'LineStyle',':');
    
    set(h2, 'LineWidth',2.0);
    set(h2, 'Color', 'red');
    set(h2, 'LineStyle','-');

    set(h1_ot, 'LineWidth',2.0);
    %set(h1_ot, 'Color', 'blue');
    h1_ot.Color =[0.0745    0.6235    1.0000];
    set(h1_ot, 'LineStyle',':');
    
    set(h2_ot, 'LineWidth',2.0);
    h2_ot.Color = [0.9804    0.3922    0.3922];
    set(h2_ot, 'LineStyle','-');
     xlim([0, 1000]);
    xticks([0:250:1200]);
    yticks(linspace(0,1,11));
    
    xlabel('2D distance [m]','fontweight', 'bold','FontSize',17,'fontname', 'times');
    
     if eps == true
        exportgraphics(gcf,'figures/los_prob_1_6m.eps');
    else
        exportgraphics(gcf,'figures/los_prob_1_6m.png','Resolution', 800);
    end
    %legend1 = legend(axes1,'show');
    %set(h,...
    %'Position',[0.625333333333333 0.011888888030582 0.244629843643423 0.0666666679382324],...
    %'NumColumns',2,...
    %'FontSize',12);

    %legend([h1, h2], 'model', 'data', ...
    %    'Location', 'northwest', 'interpreter', 'latex', 'fontsize', 10);
%subplot(1,5,2);
%nexttile;
figure();
    h3 = plot((1:length(ls_30_m))*10, ls_30_m);
    hold on;
    h4 = plot((1:length(ls_30_d))*10, ls_30_d);
    hold on;
    
    h3_ot = plot((1:length(ot_30_m))*10, ot_30_m);
    hold on;
    h4_ot = plot((1:length(ot_30_d))*10, ot_30_d);
    hold on;

    grid on;
    title('height = 30 m','FontSize',18)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 15;
    ax.XAxis.FontSize = 15;

    set(gca,'fontname','times')
    set(gca,'fontweight','bold');
    set(h3, 'LineWidth',2.0);
    set(h3, 'Color', 'blue');
    set(h3, 'LineStyle',':');
    set(h4, 'LineWidth',2.0);
    set(h4, 'Color', 'red');
    set(h4, 'LineStyle','-');

    set(h3_ot, 'LineWidth',2.0);
    %set(h1_ot, 'Color', 'blue');
    h3_ot.Color =[0.0745    0.6235    1.0000];
    set(h3_ot, 'LineStyle',':');
    
    set(h4_ot, 'LineWidth',2.0);
    h4_ot.Color = [0.9804    0.3922    0.3922];
    set(h4_ot, 'LineStyle','-');
    
    set(gca,'yticklabel',[])
    xlim([0, 1000])
    xticks([0:250:1200]);
    yticks(linspace(0,1,11));
    yticklabels([]);
    xlabel('2D distance [m]','fontweight', 'bold','FontSize',17,'fontname', 'times');
    
    if eps == true
        exportgraphics(gcf,'figures/los_prob_30m.eps');
    else
        exportgraphics(gcf,'figures/los_prob_30m.png','Resolution', 800);
    end
%subplot(1,5,3);
figure();
    h5 = plot((1:length(ls_60_m))*10, ls_60_m);
    hold on;
    h6 = plot((1:length(ls_60_d))*10, ls_60_d);
    hold on;

    h5_ot = plot((1:length(ot_60_m))*10, ot_60_m);
    hold on;
    h6_ot = plot((1:length(ot_60_d))*10, ot_60_d);
    hold on;

    grid on;
    title('height = 60 m','FontSize',18)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 15;
    ax.XAxis.FontSize = 15;

    set(gca,'fontname','times');
    set(gca,'fontweight','bold');
    set(h5, 'LineWidth',2.0);
    set(h5, 'Color', 'blue');
    set(h5, 'LineStyle',':');
    set(h6, 'LineWidth',2.0);
    set(h6, 'Color', 'red');
    set(h6, 'LineStyle','-');
    set(gca,'yticklabel',[])
        
    set(h5_ot, 'LineWidth',2.0);
    %set(h1_ot, 'Color', 'blue');
    h5_ot.Color =[0.0745    0.6235    1.0000];
    set(h5_ot, 'LineStyle',':');
    
    set(h6_ot, 'LineWidth',2.0);
    h6_ot.Color = [0.9804    0.3922    0.3922];
    set(h6_ot, 'LineStyle','-');
    %viscircles([1, 2], 3);
    xlim([0, 1000])
    xticks([0:250:1200]);
    yticks(linspace(0,1,11));
    yticklabels([]);
    xlabel('2D distance [m]','fontweight', 'bold','FontSize',17,'fontname', 'times');
    x = [0.3 0.195];
    y = [0.75 0.7];
    annotation('textarrow',x,y,'String',' LOS','FontSize',17,'FontWeight','bold','Linewidth',3)

    x = [0.7 0.815];
    y = [0.75 0.66];
    annotation('textarrow',x,y,'String','Outage','FontSize',17,'FontWeight','bold','Linewidth',3);

    if eps == true
        exportgraphics(gcf,'figures/los_prob_60m.eps');
    else
        exportgraphics(gcf,'figures/los_prob_60m.png','Resolution', 800);
    end
%subplot(1,5,4);
figure();
    h7 = plot((1:length(ls_90_m))*10, ls_90_m);
    hold on;
    h8 = plot((1:length(ls_90_d))*10, ls_90_d);
    hold on;

    h7_ot = plot((1:length(ot_90_m))*10, ot_90_m);
    hold on;
    h8_ot = plot((1:length(ot_90_d))*10, ot_90_d);
    hold on;

    grid on;
    title('height = 90 m','FontSize',18);
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 15;
    ax.XAxis.FontSize = 15;

    set(gca,'fontname','times');
    set(gca,'fontweight','bold');
    set(h7, 'LineWidth',2.0);
    set(h7, 'Color', 'blue');
    set(h7, 'LineStyle',':');
    set(h8, 'LineWidth',2.0);
    set(h8, 'Color', 'red');
    set(h8, 'LineStyle','-');
    
    set(h7_ot, 'LineWidth',2.0);
    h7_ot.Color =[0.0745    0.6235    1.0000];
    set(h7_ot, 'LineStyle',':');
    
    set(h8_ot, 'LineWidth',2.0);
    h8_ot.Color = [0.9804    0.3922    0.3922];
    set(h8_ot, 'LineStyle','-');

    set(gca,'yticklabel',[])
    xlim([0, 1000])
    xticks([0:250:1200]);
    yticks(linspace(0,1,11));
    yticklabels([]);
    xlabel('2D distance [m]','fontweight', 'bold','FontSize',17,'fontname', 'times');
    
    if eps == true
        exportgraphics(gcf,'figures/los_prob_90m.eps');
    else
        exportgraphics(gcf,'figures/los_prob_90m.png','Resolution', 800);
    end
 %subplot(1,5,5);
 figure();
    h9 = plot((1:length(ls_120_m))*10, ls_120_m);
    hold on;
    h10 =plot((1:length(ls_120_d))*10, ls_120_d);
    hold on;

    h9_ot = plot((1:length(ot_120_m))*10, ot_120_m);
    hold on;
    h10_ot = plot((1:length(ot_120_d))*10, ot_120_d);
    hold on;

    grid on;
    title('height = 120 m','FontSize',18)
    %xticks(linspace(80, 200, 7));
    ax = gca;
    ax.GridLineWidth = 2;
    ax.YAxis.FontSize = 15;
    ax.XAxis.FontSize = 15;

    set(gca,'fontname','times');
    set(gca,'fontweight','bold');
    set(h9, 'LineWidth',2.0);
    set(h9, 'Color', 'blue');
    set(h9, 'LineStyle',':');
    set(h10, 'LineWidth',2.0);
    set(h10, 'Color', 'red');
    set(h10, 'LineStyle','-');

    set(h9_ot, 'LineWidth',2.0);
    h9_ot.Color =[0.0745    0.6235    1.0000];
    set(h9_ot, 'LineStyle',':');
    
    set(h10_ot, 'LineWidth',2.0);
    h10_ot.Color = [0.9804    0.3922    0.3922];
    set(h10_ot, 'LineStyle','-');

    set(gca,'yticklabel',[]);
    xlim([0, 1000]);
    xticks([0:250:1200]);
    yticks(linspace(0,1,11));
    yticklabels([]);
    xlabel('2D distance [m]','fontweight', 'bold','FontSize',17,'fontname', 'times');

    h = legend([h9, h10], 'model', 'data', ...
    'Location', 'northeast','fontweight','bold', 'fontsize', 15,  'box','on','NumColumns', 1);


    if eps == true
        exportgraphics(gcf,'figures/los_prob_120m.eps');
    else
        exportgraphics(gcf,'figures/los_prob_120m.png','Resolution', 800);
    end
%set(gcf,'Position',[100 100 800 300]);
%exportgraphics(gcf,'figures/los_prob.png','Resolution',800);
%exportgraphics(gcf,'figures/los_prob.pdf','ContentType','vector', 'Resolution',600);
%exportgraphics(gcf,'figures/los_prob.eps');
