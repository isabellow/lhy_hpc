%% Concatenate .avi videos across sessions using ffmpeg's concat demuxer
% This does a lossless "stream copy" -- no re-encoding -- so it's fast and
% doesn't degrade video quality.
% 
% Requires ffmpeg to be installed and on your system PATH 
% (check by running `ffmpeg -version` in a terminal).
% Only works cleanly when all input videos share the same codec/resolution.

cam_ids = {'red_cam', 'yellow_cam', 'green_cam', 'blue_cam', 'bottom_cam'};
vid_roots = {
    'Z:/Isabel/data/lhy_implants/LMN88/LMN88_260722/rec_1/', ...
    'Z:/Isabel/data/lhy_implants/LMN88/LMN88_260722/rec_2/'
};
out_root = 'Z:/Isabel/data/lhy_implants/LMN88/LMN88_260722/concat_vids/';
mkdir(out_root)

for i_cam = 1:length(cam_ids)
    cam = cam_ids{i_cam};

    % full paths to this camera's video in each session, in order
    input_paths = cellfun(@(r) [r, cam, '.avi'], vid_roots, 'UniformOutput', false);
    output_path = [out_root, cam, '.avi'];

    % ffmpeg's concat demuxer takes a text file listing the inputs, one per
    % line, formatted as: file 'path/to/video.avi'
    % write this list to a temp file (one per camera, so parallel loop
    % iterations -- if you ever add any -- wouldn't clobber each other)
    list_path = [out_root, cam, '_concat_list.txt'];
    fid = fopen(list_path, 'w');
    for i_input = 1:length(input_paths)
        fprintf(fid, "file '%s'\n", input_paths{i_input});
    end
    fclose(fid);

    % -f concat            : use the concat demuxer
    % -safe 0              : allow absolute file paths in the list (ffmpeg
    %                        is cautious by default about paths outside the
    %                        list file's own directory)
    % -i list_path          : the input is the list file itself
    % -c copy               : "copy" the audio/video streams as-is instead
    %                        of decoding + re-encoding -- this is what makes
    %                        it lossless and fast
    cmd = sprintf('ffmpeg -f concat -safe 0 -i "%s" -c copy "%s"', list_path, output_path);
    [status, cmd_output] = system(cmd);

    if status ~= 0
        % status is ffmpeg's exit code -- nonzero means it failed (e.g.
        % mismatched codecs between inputs, or ffmpeg not found on PATH)
        error('ffmpeg failed for %s:\n%s', cam, cmd_output)
    end

    fprintf('%s: concatenated -> %s\n', cam, output_path)
    delete(list_path) % clean up the temp list file

    % sanity check: output frame count should equal the sum of inputs frame counts
    reader = VideoReader(output_path);
    n_frames_out = reader.NumFrames;
    n_frames_in = 0;
    for i_path = 1:length(input_paths)
        reader = VideoReader(input_paths{i_path});
        n_frames_in = n_frames_in + reader.NumFrames;
    end
    fprintf('  inputs total frame count: %d\n', n_frames_in)
    fprintf('  output frame count: %d\n', n_frames_out)
    if n_frames_out ~= sum(n_frames_in)
        warning('frame count mismatch!')
    end
end