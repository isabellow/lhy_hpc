%% Load the arena interactions and other components needed to run the site interaction GUI
clear all
close all

%% Set paths and load files
% functions to run site interaction GUI
addpath(genpath('C:\Users\ilow1\Documents\code\lhy_hpc\behavior\')) % GUI functions

% root session directory
sessionPath = 'Z:\Isabel\data\hpc_implants\RBY94\RBY94_241125\';
cd(sessionPath),

% cache locations and videos
load('cacheLoc_botCam.mat'),
botVid = fullfile(sessionPath,'bottom_cam.avi'); %VideoReader('botCam.avi');
topVid = fullfile(sessionPath,'blue_cam.avi'); %VideoReader('lFront.avi');

% ffmpegWrapper = 'C:\Users\ilow1\Documents\code\VideoReaderFFMPEG';
% addpath(genpath(ffmpegWrapper))
% ffmpegPath = 'C:\ffmpeg\bin';
% botVid = VideoReaderFFMPEG('bottom_cam.avi','FFMPEGPATH',ffmpegPath);
% topVid = VideoReaderFFMPEG('blue_cam.avi','FFMPEGPATH',ffmpegPath);

% Define the cacheNet
cacheNet_root = 'C:\Users\Isabel\Documents\code\bird_pose_tracking\cacheNet\';
cacheNet_file = 'cacheNet_739646_6829';
cacheNet = load(fullfile(cacheNet_root, cacheNet_file));
cacheNet = cacheNet.cacheNet;

%% Load and update the seed struct
load('behavior_data\seed_struct.mat')

% reformat indices for matlab
seedStruct.countData.siteNum = seedStruct.countData.siteNum + 1;
seedStruct.countData.feederNum = seedStruct.countData.feederNum + 1;
seedStruct.countData.waterNum = seedStruct.countData.waterNum + 1;
seedStruct.countData.beakPerchNum = seedStruct.countData.beakPerchNum + 1;

% transpose 1D arrays and convert to doubles as-needed
fields = fieldnames(seedStruct); % Get all field names
for i = 1:numel(fields)
    field = fields{i};
    array = seedStruct.(field);
    
    % Check if the field is a numeric matrix
    if isnumeric(array)
        % Convert to double and transpose
        if length(array) > 1
            array = array.';
        end
        seedStruct.(field) = double(array); 
    end
end

fields = fieldnames(seedStruct.countData); % Get all field names
for i = 1:numel(fields)
    field = fields{i};
    array = seedStruct.countData.(field);
    
    % Check if the field is a numeric matrix
    if isnumeric(array)
        % Convert to double and transpose
        array = array.';
        seedStruct.countData.(field) = double(array); 
    end
end

% add cache locations
seedStruct.cacheLoc = cacheLocBot_distort;

%% Run the GUI
% view on the L is the pre and post image of the current site
% graph on the R is the seed count 
% most useful workflow is to go to the first flag
% then, toggle to Site # to see all the interactions for that site.
% The hot key short-cut for this is "n"
seedApp = siteInteractionGUI(seedStruct, botVid, cacheNet, topVid);


% add to GUI a plot of the 'smoothSeed' trace aligned to site interaction,
% with onset/offset of interaction and any gain/loss marked


%% Possibly these things can be moved elsewhere?
load('annotatedSeeds.mat')
makeEthogramPlot(annotatedSeeds,smPtsVel, [10,22;54,64;124,130], 1e3);

%% use GUI results to update seed event tracking
% get gain times and retrieval site interactions
[gainCacheInt,gainCacheSite] = find(seedStruct.seedChanges==-1);
gainTimesInt = seedStruct.newCacheTimes(gainCacheInt)/2 + seedStruct.endCacheTimes(gainCacheInt)/2;
gainTimesEvent = mean(seedStruct.seedTimes(:,1:2),2);

% get loss times and cache site interactions
[loseCacheInt,loseCacheSite] = find(seedStruct.seedChanges==1);
loseTimesInt = seedStruct.newCacheTimes(loseCacheInt)/2 + seedStruct.endCacheTimes(loseCacheInt)/2;
loseTimesEvent = mean(seedStruct.seedTimes(:,3:4),2);

% find closest gainTime for each retrieval interaction, and sanity check
% that interaction is further from any lossTime than this gainTime
[gainVal, gainInd] = min(abs(gainTimesInt'-gainTimesEvent));
gainThreshs = min(abs(gainTimesInt'-loseTimesEvent));
validGain = gainVal<gainThreshs;

% same for loss
[loseVal, loseInd] = min(abs(loseTimesInt'-loseTimesEvent));
loseThreshs = min(abs(loseTimesInt'-gainTimesEvent));
validLoss = loseVal<loseThreshs;

% get eventNum and siteNum for valid gains/retrievals, and loss/caches
retEvent = gainInd(validGain);
retEventSite = gainCacheSite(validGain);
cacheEvent = loseInd(validLoss);
cacheEventSite = loseCacheSite(validLoss);
%%

footPos = mean(smPts(:,:,[13,18]),3);
footPosThresh = 0.1; % this is less than minimum distance btw sites (>0.1)
nEvents = size(seedStruct.seedTimes,1);
nSites = length(seedStruct.initSeedCounts);
seedID = 0;
seedEventIDs = nan(nEvents,2); % log seedID for each gain/loss
cacheIDs = cell(nEvents,nSites,2); % each site for each event, before gain and before loss
% assign an ID to each seed in initially occupied sites
occSites = find(seedStruct.initSeedCounts);
for nSite = occSites(:)'
    nSeeds = seedStruct.initSeedCounts(nSite);
    for nSeed = 1:nSeeds
        seedID = seedID + 1; % increment the seed ID
%         cacheIDs{1,nSite,1} = cat(1, cacheIDs{1,nSite,1}, seedID);
        cacheIDs{1,nSite,1}(end+1) = seedID;
    end
end

% track seedIDs across seed-carrying events
%TODO: handle start/end edge cases?
for nEvent = 1:nEvents
    % handle seed tracking for retrieval and other seedGains
    isRet = ismember(nEvent, retEvent);
    if isRet % if retrieval, get last cache at this site as ID
        retSite = retEventSite(retEvent==nEvent);
        thisSeed = cacheIDs{nEvent, retSite, 1}(end); % this should be most recently cached seedID
        % copy pre-gain to pre-loss cell array, then remove the gained seed
        cacheIDs(nEvent, :, 2) = cacheIDs(nEvent, :, 1);
        cacheIDs{nEvent, retSite, 2}(end) = [];
    else
        % cacheIDs are unchanged
        cacheIDs(nEvent, :, 2) = cacheIDs(nEvent, :, 1);
        % check if we need a new seedID, or if feet have not moved use the
        % previous seedID (bird was eating or momentarily lost tracking)
        footPosGain = footPos(seedStruct.seedTimes(nEvent,2),:);
        if nEvent>1
            footPosLastLoss = footPos(seedStruct.seedTimes(nEvent-1,4),:);
        else % if this is the first event, make distance infinite to force a new seedID
            footPosLastLoss = inf.*[1,1,1];
        end
        footPosDiff = sqrt(sum((footPosGain-footPosLastLoss).^2,2));
        % if feet are in new place, or last loss was a cache (since
        % by definition this is not a retrieval!), this is a new seed
        if footPosDiff>footPosThresh || ismember(nEvent-1, cacheEvent)
            seedID = seedID + 1; % increment the seedID count
            thisSeed = seedID; % current seed is the newest ID
        end
        % NOTE: when this is not a new seed, we do nothing, because current
        % seed has not changed
    end
    seedEventIDs(nEvent, 1) = thisSeed;
    
    % handle seed tracking for caches and other seedLosses
    isCache = ismember(nEvent, cacheEvent);
    if isCache % if this is a cache, add seed to cacheIDs
        cacheSite = cacheEventSite(cacheEvent==nEvent);
        % copy pre-loss to next event's pre-gain (i.e. this post-loss),
        % then add the lost seed to this site
        cacheIDs(nEvent+1, :, 1) = cacheIDs(nEvent, :, 2);
%         cacheIDs{nEvent+1, cacheSite, 1} = cat(1, cacheIDs{nEvent+1, cacheSite, 1}, seedID);
        cacheIDs{nEvent+1, cacheSite, 1}(end+1) = seedID;
        thisSeed = 0;
    else
        % if it's not a cache, cacheIDs are unchanged
        cacheIDs(nEvent+1, :, 1) = cacheIDs(nEvent, :, 2);
        % don't change thisSeed, as it may be regained next
    end
    seedEventIDs(nEvent, 2) = thisSeed;
end

% build a "seed story" by iterating through curated results from GUI. 
% When first seed is gained, create a new 'seed ID'. Record where this ID
% is next lost. If it's lost to cache site, add it to the site count. When
% future seed is gained, check if previous seed has been lost at this
% location. Match w/ the most recent loss to this site, if any. Some
% caution here: exclude central feeder, also be wary of loss-gains from
% perch that are not directly sequential? If there is no previous loss here
% or pickup is not from a site/perch, create a new seed ID for this gain!

% to detect where seed is gained/lost from, use results from manual GUI for
% sites, modify the simpler fully-automatic detection from old GUI; see if
% at any time point within a lastLose-Gain or lastGain-Lose beak is close to a
% perch. If it is, and the gain/loss is not already "claimed" by a cache
% site, then assign it to the perch.

% Store "seed stories" as a timeseries where each change in seed location
% (i.e. all gains and losses) indicate the seed ID and the location. It may
% also be useful to build a cell array at each of these timepoints,
% listing ID #s of seeds in each site (or perch) after that seed-change.
% Seed count is then the length of each list in the cell array


