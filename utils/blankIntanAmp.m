function [h] = blankIntanAmp(filepath, amp_file, h, tduration, tstart, buffer)
%BLANKINTANAMP overwrites data in an Intan Technologies RHD2000 amplifier data file.
%
%   [h] = BLANKINTANAMP(filepath, amp_file, h, tduration, tstart) overwrites the
%   amplifier data for a specified duration (in seconds) starting from
%   the specified start time (in seconds) with zeros and smoothes the edges.
%
%   Inputs:
%       - filepath              : String specifying the path to the data folder.
%                                 If not provided, a dialog will prompt the user.
%       - amp_file              : String specifying the amplifier data file.
%                                 If not provided, uses 'amplifier_copy.dat'.
%       - h                     : Struct containing header information from the info.rhd file.
%                                 If not provided, loads the header from the data folder.
%       - tduration             : Duration in seconds to read from the file.
%                                 If not provided, the entire file is read.
%       - tstart                : Start time in seconds from the beginning of the file.
%                                 If not provided, reading starts from the beginning.
%       - buffer                : Number of samples over which to smooth the signal to zero.
%                                 If not provided, default is 1 ms worth of samples.
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

% SET VARIABLES
if ~exist('filepath','var')
    [filepath] = ...
        uigetdir('Select a folder with single file per data type');
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

% GET THE RAW DATA
% Read out the data from the desired window
raw_data = fread(fid, [nchannels, num_samples], 'int16');
raw_data_dbl = double(raw_data);

% Rewind to the correct start time
if exist('tstart','var') && ~isempty(tstart)
    fseek(fid, ind_start*2*nchannels, 'bof');
else
    frewind(fid);
end

% INTERPOLATE AND SMOOTH
% Define a window for convolution
smooth = 7;
w = gausswin(smooth);
w = w / sum(w);
half_s = floor(smooth/2);

% Define a time window for interpolation
start_idx = [1:buffer];
end_idx = [num_samples-buffer+1:num_samples];
ds_idx = [buffer+1, num_samples-buffer];
full_idx = [buffer+1:num_samples-buffer];

% Interpolate between the raw data abutting the blank period + smooth edges
blank_data = zeros(nchannels, num_samples);
for c = 1:nchannels
    % match ends to data + interpolate
    raw_data_ch = raw_data_dbl(c, :);
    ds_data = raw_data_ch(ds_idx);
    interp_data = interp1(ds_idx, ds_data, full_idx, 'linear');
    
    % smooth everything
    smooth_data = [raw_data_ch(start_idx), [interp_data, raw_data_ch(end_idx)]];
    to_interp = conv(smooth_data, w, 'valid');
    smooth_data(half_s+1:end-half_s) = to_interp;
    
    blank_data(c, :) = smooth_data;
    
    % sanity check
    if c == 1 %&& show_plot
        hold on
        % plot(raw_data_ch)
        % plot(full_idx, interp_data)
        plot(blank_data(c, :))
    end
end

% Replace in the original file
blank_data = int16(blank_data);
fwrite(fid, blank_data, 'int16', 'l'); 

fclose(fid);
