function [hFig,hAx1,hAx2] = makeEthogramPlot(annotatedSeeds,smPtsVel, fOpen, dsFac)
    arguments
        annotatedSeeds struct
        smPtsVel (:,3,18) double
        fOpen (:,2) double = [5,15; 65,70] % feeder open times
        dsFac (1,1) double = 600 % ethogram smoothing size, in samples at 60hz
    end
validInd = annotatedSeeds.validFrames;
countData = annotatedSeeds.countData;
fps = 50; % Hz
norm_unit = 13 * 2.54; % arena half-width in cm

%% use image patch to represent feeder open times

figure, hFig=tiledlayout(2,1);
hFig.TileSpacing = 'compact'; hFig.Padding='compact';
nexttile,hold on
for i=1:size(fOpen,1)
    v = [fOpen(i,1),0; fOpen(i,1),1;...
        fOpen(i,2),0; fOpen(i,2),1];
    f = [1,2,4,3];
    patch('Faces',f,'Vertices',v,'FaceColor',[0,0,0],'FaceAlpha',0.2,'EdgeColor','none');
end

for i=1:length(countData.newBeakPerch)
    v = [countData.newBeakPerch(i),0; countData.newBeakPerch(i),.975;...
        countData.endBeakPerch(i),0; countData.endBeakPerch(i),.975];
    v(:,1) = v(:,1)/(fps*60); % convert to minutes
    f = [1,2,4,3];
    patch('Faces',f,'Vertices',v,'FaceColor',[.47,.68,.19],'FaceAlpha',0.5,'EdgeColor','none');
end

for i=1:length(countData.newFeeder)
    v = [countData.newFeeder(i),0; countData.newFeeder(i),.95;...
        countData.endFeeder(i),0; countData.endFeeder(i),.95];
    v(:,1) = v(:,1)/(fps*60);
    f = [1,2,4,3];
    patch('Faces',f,'Vertices',v,'FaceColor',[.85,.33,.1],'FaceAlpha',0.5,'EdgeColor','none');
end

for i=1:length(countData.newWater)
    v = [countData.newWater(i),0; countData.newWater(i),1;...
        countData.endWater(i),0; countData.endWater(i),1];
    v(:,1) = v(:,1)/(fps*60);
    f = [1,2,4,3];
    patch('Faces',f,'Vertices',v,'FaceColor',[.0,.45,.75],'FaceAlpha',0.5,'EdgeColor','none');
end

axis tight,
%% activity level, measured by body speed
tmp = squeeze(sqrt(sum(smPtsVel.^2,2)));
tmp(~validInd,:) = nan;
tmp = movmean(tmp,dsFac,'omitnan');
% dsSpd = mean(tmp(:,[5,6,7,11,16]),2);
dsSpd = mean(tmp(:,[13,18]),2);
dsSpd = dsSpd * norm_unit * fps; % convert to cm/s
plot((1:length(dsSpd))/(fps*60),dsSpd/max(dsSpd),'k'),
ylabel('Movement Speed (cm/s)'),
y_ticks = round(linspace(0,max(dsSpd),3));
yticks([0,.5,1]),
yticklabels(num2cell(y_ticks)),
hAx1 = gca;
ax = xlim;
%% calculate rates of new perch visits and site interactions
seedCount = cumsum(annotatedSeeds.seedChanges)+annotatedSeeds.initSeedCounts';
preIntSeedCount = seedCount - annotatedSeeds.seedChanges;
siteOcc = false(length(countData.newSite),1);
for i=1:length(siteOcc)
    siteOcc(i) = preIntSeedCount(i,countData.siteNum(i))>0;
end
visitsNoInt = ~any( countData.newPerch<=countData.newSite' & ...
    countData.endPerch>=countData.newSite', 2);
siteIntNoChange = all(annotatedSeeds.seedChanges==0,2);
tmp = zeros(length(dsSpd),3);
tmp(countData.newPerch(visitsNoInt), 1) = 1;
tmp(countData.newSite(siteIntNoChange & ~siteOcc), 2) = 1;
tmp(countData.newSite(siteIntNoChange & siteOcc), 3) = 1;
dsEventRate = movmean(tmp,dsFac)*(fps*60);
% figure,plot((1:length(dsSpd))/(fps*60),dsEventRate),

%% calculate arena seed count, cumulative caches, cumulative retrievals

siteIntCache = cumsum(any(annotatedSeeds.seedChanges>0,2));
siteIntRet = cumsum(any(annotatedSeeds.seedChanges<0,2));
nexttile,yyaxis left,
a=area((1:length(dsSpd))/(fps*60),dsEventRate(:,[3,2]),'LineStyle','none');
a(2).FaceColor = 'k';
ylim([0, max(sum(dsEventRate(:,2:3),2))]),
% hold on, plot((1:length(dsSpd))/60^2,dsEventRate(:,2),'k-','linewidth',1),
% plot((1:length(dsSpd))/60^2,dsEventRate(:,3),'-','linewidth',1),
ylabel('Checks per Min'),
yyaxis right, hold on,
plot(annotatedSeeds.newCacheTimes/(fps*60),siteIntCache,'-','linewidth',2,'color',[.85,.33,.1])
plot(annotatedSeeds.newCacheTimes/(fps*60),siteIntRet,'-','linewidth',2,'color',[.45,.67,.2])
ylim([0, 1+max([siteIntCache(end),siteIntRet(end)])]),
ylabel('Cumulative Caches/Retrievals'),
% axis tight,
hAx2 = gca;
xlabel('Time (min)'),
linkaxes([hAx1,hAx2],'x'),
xlim(ax),