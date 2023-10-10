function [dig_data, h] = readIntanDig(filepath, tduration, tstart)
%READINTANDIG Reads Intan Technologies RHD2000 digital data file.
%
%   [dig_data, h] = READINTANDIG(filepath) reads the digital data from the
%   specified filepath. If filepath is not provided, a dialog will prompt
%   the user to select a folder. The function returns the raw digital data
%   in matrix form (n channels x n samples) and the header information.
%
%   [dig_data, h] = READINTANDIG(filepath, tduration) reads the digital
%   data for a specified duration (in seconds) from the start of the file.
%
%   [dig_data, h] = READINTANDIG(filepath, tduration, tstart) reads the
%   digital data for a specified duration (in seconds) starting from the
%   specified start time (in seconds).
%
%   Inputs:
%       - filepath : String specifying the path to the data folder.
%                    If not provided, a dialog will prompt the user.
%       - tduration: Duration in seconds to read from the file.
%                    If not provided, the entire file is read.
%       - tstart   : Start time in seconds from the beginning of the file.
%                    If not provided, reading starts from the beginning.
%
%   Outputs:
%       - dig_data : Matrix containing the raw digital data.
%                    Size is (n channels x n samples).
%       - h        : Struct containing header information from the info.rhd file.
%
%   Example:
%       [data, header] = readIntanDig('D:\data\HC15_231007\raw_231007_125517', 10, 5);
%       This reads 10 seconds of data starting from the 5th second.
%
%   Note:
%       This function is based on read_Intan_RHD2000_file and has been
%       edited by Hannah Payne, Aronov lab.
%
%   See also: READINTANINFO, READINTANAMP

if ~exist('filepath','var')
    [filepath] = ...
        uigetdir('Select a folder with single file per data type data');
    if (filepath == 0);  return; end
    
end

% Open the info file
info_filepath = fullfile(filepath,'info.rhd');
fprintf('Opening info file %s\n',info_filepath)
h = readIntanInfo(info_filepath);
ndig_in = h.num_h.board_dig_in_channels;
dig_data = [];

if ndig_in
    % Open the digital data
    dig_filepath = fullfile(filepath,'digitalin.dat');
    fid = fopen(dig_filepath, 'r');
    
    % Get the total number of time point samples
    fileinfo = dir(dig_filepath);
    num_samples = fileinfo.bytes/2; % uint16 = 2 bytes
    
    % Change start time if needed
    if exist('tstart','var') && ~isempty(tstart)
        ind_start = round(tstart*h.sample_rate);
        fseek(fid, ind_start*2, 'bof');
        num_samples = num_samples - ind_start;
    end
    
    % Change the duration to read if needed
    if exist('tduration','var') && ~isempty(tduration)
        num_samples = round(tduration*h.sample_rate);
    end
    
    digital_word = fread(fid, num_samples, 'uint16');
    fclose(fid);
    
    % Individual digital inputs can be isolated using the bitand function in MATLAB:
    % digital_input_ch = (bitand(digital_word, 2^ch) > 0); % ch has a value of 0-15 here
    dig_data = false(ndig_in, length(digital_word));
    for ii = 1:ndig_in
        ch = h.board_dig_in_channels(ii).native_order;
        dig_data(ii,:) = (bitand(digital_word, 2^ch) > 0);
    end
end