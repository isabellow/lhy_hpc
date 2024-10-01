function [h] = replaceDataIntanAmp(filepath, amp_file, method,...
                                    h, tduration, tstart)
%REPLACEDATAINTANAMP overwrites data in an Intan Technologies RHD2000 amplifier data file.
%
%   [h] = REPLACEDATAINTANAMP(filepath, amp_file, h, tduration, tstart) overwrites the
%   amplifier data for a specified duration (in seconds) starting from
%   the specified start time (in seconds) with zeros.
%
%   Inputs:
%       - filepath              : String specifying the path to the data folder.
%                                 If not provided, a dialog will prompt the user.
%       - amp_file              : String specifying the amplifier data file.
%                                 If not provided, uses 'amplifier_copy.dat'.
%       - method                : String specifiying the overwrite intention.
%                                 'blank' replaces the data chunk with zeros.
%                                 'smooth' passes a gaussian filter over the data.
%       - h                     : Struct containing header information from the info.rhd file.
%                                 If not provided, loads the header from the data folder.
%       - tduration             : Duration in seconds to read from the file.
%                                 If not provided, the entire file is read.
%       - tstart                : Start time in seconds from the beginning of the file.
%                                 If not provided, reading starts from the beginning.
%
%   Outputs:
%       - h             : Struct containing header information from the info.rhd file.
%
%   Example:
%       header = replaceDataIntanAmp('D:\data\HC15_231007\raw_231007_125517', 
%                                       'amplifier_copy.dat', 'blank', header, 10, 5);
%       This overwrites 10 seconds of data starting from the 5th second
%       with zeros.
%
%   Note:
%       This function is originally based on read_Intan_RHD2000_file.
%       It was edited by Hannah Payne and subsequently Isabel Low, Aronov lab.
%
%   See also: READINTANAMP, READINTANINFO, UIGETDIR, FWRITE, 

if ~exist('filepath','var')
    [filepath] = ...
        uigetdir('Select a folder with single file per data type data');
    if (filepath == 0);  return; end
end
if ~exist('amp_file','var')
    amp_file = 'amplifier_copy.dat';
end

% Open the info file
if ~exist('h','var')
    info_filepath = fullfile(filepath,'info.rhd');
    fprintf('Opening info file %s\n',info_filepath)
    h = readIntanInfo(info_filepath);
end
nchannels = h.num_h.amplifier_channels;

% Open the amplifier data (read/write access)
amp_filepath = fullfile(filepath, amp_file);
fid = fopen(amp_filepath,'r+');

% Get the total number of time point samples
fileinfo = dir(amp_filepath);
num_samples = fileinfo.bytes/(nchannels * 2); % int16 = 2 bytes

% Change start time if needed
if exist('tstart','var') && ~isempty(tstart)
    ind_start = round(tstart*h.sample_rate);
    fseek(fid, ind_start*2*nchannels, 'bof');
    num_samples = num_samples - ind_start;
end

% Change the duration to overwrite if needed
if exist('tduration','var') && ~isempty(tduration)
    num_samples= round(tduration*h.sample_rate);
end

% Overwrite the data with the desired change
if strcmp(method, 'blank')
    % Blank the data in the desired time range
    blank_data = zeros(1, num_samples*nchannels, 'int16');
    fwrite(fid, blank_data, 'int16');

elseif strcmp(method, 'smooth')
    % Read out the data from the desired window and smooth it
    raw_data = fread(fid, [nchannels, num_samples], 'int16')*0.195;
    w = gausswin(10);
    w = w / sum(w);
    smooth_data = zeros(nchannels, num_samples, 'int16');
    for c = 1:nchannels
        smooth_data(c, :) = conv(raw_data(c, :), w, 'same');
    end
    
    % Rewind to the correct start time
    if exist('tstart','var') && ~isempty(tstart)
        fseek(fid, ind_start*2*nchannels, 'bof');
    else
        frewind(fid);
    end
    
    % Replace in the original file
    fwrite(fid, smooth_data, 'int16');    

else
    fprintf('method not recognized! enter blank or smooth\n')
end

fclose(fid);
