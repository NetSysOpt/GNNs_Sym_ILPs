clc;
clear all;
close all;

pathTemp = 'exp/dataset-%s-Aug-%s-opt-opt-epoch-100-sampleTimes-%s/loss_record.mat';
sampleTimes = '8';
markerInds = [20,40,60,80,100];
xticks = [20,40,60,80,100];
sw = 10;
figure('Position', [0, 0,1300,400]);
t = tiledlayout(1,3,"TileSpacing","compact","Padding","tight");

% BPP
dataset='BPP';
gaussian = load(sprintf(pathTemp,dataset,'uniform',sampleTimes));
pos = load(sprintf(pathTemp,dataset,'pos',sampleTimes));
orbit = load(sprintf(pathTemp,dataset,'orbit',sampleTimes));
group = load(sprintf(pathTemp,dataset,'group',sampleTimes));

loss_scale = 100; % normalize 
dataset_name = 'BPP';
nexttile;
drawFigure('valid','Valid loss',gaussian,pos,orbit,group,markerInds,xticks,sw, dataset_name, loss_scale);
ylim([0.19, 0.4]);

%%BIP
dataset='BIP';
gaussian = load(sprintf(pathTemp,dataset,'uniform',sampleTimes));
pos = load(sprintf(pathTemp,dataset,'pos',sampleTimes));
orbit = load(sprintf(pathTemp,dataset,'orbit',sampleTimes));
group = load(sprintf(pathTemp,dataset,'group',sampleTimes));

loss_scale = 1000; % normalize 
dataset_name = 'BIP';
nexttile;
drawFigure('valid','Valid loss',gaussian,pos,orbit,group,markerInds,xticks,sw, dataset_name,loss_scale);
ylim([0.26, 0.4]);


%SMSP
dataset='SMSP';
gaussian = load(sprintf(pathTemp,dataset,'uniform',sampleTimes));
pos = load(sprintf(pathTemp,dataset,'pos',sampleTimes));
orbit = load(sprintf(pathTemp,dataset,'orbit',sampleTimes));
group = load(sprintf(pathTemp,dataset,'group',sampleTimes));

dataset_name = 'SMSP';
loss_scale = 1000; % normalize 
nexttile;
drawFigure('valid','Valid loss',gaussian,pos,orbit,group,markerInds,xticks,sw, dataset_name, loss_scale);
ylim([0.55,1]);


leg = legend('Uniform','Position ','Orbit','Orbit+','Color',[0.99,0.995,1]);
set(leg,'Interpreter','latex','FontSize',20, 'Location','best','ItemTokenSize',[50,20],'Orientation','horizontal', ...
    "NumColumns",4);

leg.Layout.Tile = 'north';
set(gcf,'PaperType','a3');

plot_path = '.\';
plot_name = 'validation_loss';
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

code_end = 1;

function drawFigure(mod,tit_name,gaussian,pos,orbit,group,markerInds,xticks,sw, dataset_name, loss_scale)
%     figure;
    ss = size(gaussian.train_loss);
    epoch = 1:ss(2);
    linwidth = 2;
    blue = [0.00,0.45,0.74];
    orange = [0.85,0.33,0.10];
    red = [0.850980392156863 0.325490196078431 0.0980392156862745];
    green = [0.466666666666667 0.674509803921569 0.188235294117647];
    purple = [0.494117647058824 0.184313725490196 0.556862745098039];
    title(dataset_name,'Interpreter','latex', 'FontSize', 18);hold on;
        
    loss1 = gaussian.(strcat(mod,'_loss'));
    loss1 = smooth_v(loss1,sw, loss_scale);
    plot(epoch,loss1,'LineWidth',linwidth,'Marker','*','MarkerIndices',markerInds,'MarkerSize',13,'Color',blue);hold on;
    
    loss2 = pos.(strcat(mod,'_loss'));
    loss2 = smooth_v(loss2,sw, loss_scale);
    plot(epoch,loss2,'LineWidth',linwidth,'Marker','o','MarkerIndices',markerInds,'MarkerSize',13,'Color',red);hold on;
    
    loss3 = orbit.(strcat(mod,'_loss'));
    loss3 = smooth_v(loss3,sw, loss_scale);
    plot(epoch,loss3,'LineWidth',linwidth,'Marker','square','MarkerIndices',markerInds,'MarkerSize',13,'Color',green);hold on;
    
    loss4 = group.(strcat(mod,'_loss'));
    loss4 = smooth_v(loss4,sw, loss_scale);
    plot(epoch,loss4,'LineWidth',linwidth,'Marker','diamond','MarkerIndices',markerInds,'MarkerSize',13,'Color',purple);hold on;
    
    xlabel('Epoch','Interpreter','latex','FontSize',18);
    ylabel('Loss','Interpreter','latex','FontSize',18);
    set(gca,'FontSize',15,'XTick',xticks);

end

% smooth the curve
function sv = smooth_v(v,w, loss_scale)
    v_len = size(v);
    v_len = v_len(2);
    sv = 1:v_len;
    tv = 1:v_len+2*(w-1);
    for i=1:v_len
        sv(i) = v(i);
        tv(i+w-1) = v(i);
    end
    for i=1:w-1
        tv(v_len+w-1+i) = v(v_len);
        tv(i) = v(1);
    end

    for i=1:v_len
        ss = sv(i);
        for j=1:w-1
            ss = ss + tv(w-1+i+j) + tv(w-1+i-j);
        end
        ss = ss/(2*(w-1)+1);
        sv(i) = ss;
    end

    sv = sv/loss_scale;
end