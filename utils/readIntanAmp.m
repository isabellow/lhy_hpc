function [amplifier_data, h] = readIntanAmp(filepath, tduration, tstart)
%READINTANAMP Reads Intan Technologies RHD2000 amplifier data file.
%
%   [amplifier_data, h] = READINTANAMP(filepath) reads the amplifier data
%   from the specified filepath. If filepath is not provided, a dialog will
%   prompt the user to select a folder. The function returns the raw
%   amplifier data in matrix form (n channels x n samples) and the header
%   information.
%
%   [amplifier_data, h] = READINTANAMP(filepath, tduration) reads the
%   amplifier data for a specified duration (in seconds) from the start of
%   the file.
%
%   [amplifier_data, h] = READINTANAMP(filepath, tduration, tstart) reads
%   the amplifier data for a specified duration (in seconds) starting from
%   the specified start time (in seconds).
%
%   Inputs:
%       - filepath              : String specifying the path to the data folder.
%                                 If not provided, a dialog will prompt the user.
%       - tduration             : Duration in seconds to read from the file.
%                                 If not provided, the entire file is read.
%       - tstart                : Start time in seconds from the beginning of the file.
%                                 If not provided, reading starts from the beginning.
%
%   Outputs:
%       - amplifier_data: Matrix containing the raw amplifier data.
%                         Size is (n channels x n samples).
%       - h             : Struct containing header information from the info.rhd file.
%
%   Example:
%       [data, header] = readIntanAmp('D:\data\HC15_231007\raw_231007_125517', 10, 5);
%       This reads 10 seconds of data starting from the 5th second
%
%   Note:
%       This function is originally based on read_Intan_RHD2000_file.
%       It was edited by Hannah Payne and subsequently Isabel Low, Aronov lab.
%
%   See also: READINTANINFO, UIGETDIR, FREAD, 

if ~exist('filepath','var')
    [filepath] = ...
        uigetdir('Select a folder with single file per data type data');
    if (filepath == 0);  return; end
end

% Open the info file
info_filepath = fullfile(filepath,'info.rhd');
fprintf('Opening info file %s\n',info_filepath)
h = readIntanInfo(info_filepath);
nchannels = h.num_h.amplifier_channels;

% Open the amplifier data
amp_filepath = fullfile(filepath,'amplifier.dat');
% amp_filepath = fullfile(filepath,'amplifier_stim_blanked.dat');
fid = fopen(amp_filepath,'r');

% Get the total number of time point samples
fileinfo = dir(amp_filepath);
num_samples = fileinfo.bytes/(nchannels * 2); % int16 = 2 bytes

% Change start time if needed
if exist('tstart','var') && ~isempty(tstart)
    ind_start = round(tstart*h.sample_rate);
    fseek(fid, ind_start*2*nchannels, 'bof');
    num_samples = num_samples - ind_start; % CHECK THIS
end

% Change the duration to read if needed
if exist('tduration','var') && ~isempty(tduration)
    num_samples = round(tduration*h.sample_rate);
end

% Read the data
% To convert to electrode voltage in microvolts, multiply by 0.195
amplifier_data = fread(fid, [nchannels, num_samples], 'int16')*0.195;
fclose(fid);


