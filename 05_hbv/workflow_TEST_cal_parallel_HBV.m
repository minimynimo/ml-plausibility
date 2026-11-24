function workflow_TEST_cal_parallel_HBV(start_num, end_num, basin_list_file, out_root, expected_data_start_str, cal_start_date_str, cal_end_date_str, optimiser_seed, max_iter, mat_root)
% HBV (MARRMoT m_37_hbv_15p_5s) per-basin calibration + scenario runs for the HSP study.
% All dates/seeds/paths come from config/experiment.py via 05_hbv/parallel.sh.
%
%   start_num, end_num      : 1-based indices into the basin list (one MATLAB process per range)
%   basin_list_file         : text file, one gauge id per line (the training pool)
%   out_root                : output directory (Q/, Parameters/, calib_log/, failed/, cmaes_output/)
%   expected_data_start_str : first date the .mat files must start on (asserted, never assumed)
%   cal_start/cal_end       : calibration window (KGE), e.g. 1988-01-01 .. 1997-12-31
%   optimiser_seed          : CMA-ES random seed
%   max_iter                : CMA-ES MaxIter (optional, default 300; small values for smoke tests only)
%   mat_root                : folder with the MARRMoT_InputData_US_v212{,_no_precip,_up} .mat folders
%                             (optional, default <repo>/data/hbv_mat; built by 01_data_prep/5_create_mat_files.py)
%
% The simulation always starts on the first date of the .mat file with empty stores (S0 = 0), so
% the days before cal_start (1987) are the warm-up. The start date is read from the file's
% dates_as_datenum (real MATLAB datenums) and asserted equal to expected_data_start_str.
%
% Outputs per basin (written only after every scenario ran):
%   Q/varssim_<id>_Q.csv              Date + one column per scenario (og, no_precip, peak*_rate*, nostorm)
%   Parameters/varssim_<id>_calibrated_parameters.csv
%   calib_log/varssim_<id>_calib.csv  KGE_cal, n_evals, stopflag, optimiser settings, seed, runtime
%   failed/varssim_<id>_failed.txt    reason, if anything failed

if nargin < 9 || isempty(max_iter), max_iter = 300; end
script_dir = fileparts(mfilename('fullpath'));
if nargin < 10 || isempty(mat_root), mat_root = fullfile(script_dir, '..', 'data', 'hbv_mat'); end
expected_data_start = datetime(expected_data_start_str, 'InputFormat', 'yyyy-MM-dd');
cal_start_date  = datetime(cal_start_date_str,  'InputFormat', 'yyyy-MM-dd');
cal_end_date    = datetime(cal_end_date_str,    'InputFormat', 'yyyy-MM-dd');
fprintf('Expected data start: %s | Calibration: %s to %s | CMA-ES seed %d\n', ...
    datestr(expected_data_start), datestr(cal_start_date), datestr(cal_end_date), optimiser_seed);

input_dir           = fullfile(mat_root, 'MARRMoT_InputData_US_v212');
no_precip_input_dir = fullfile(mat_root, 'MARRMoT_InputData_US_v212_no_precip');
up_input_dir        = fullfile(mat_root, 'MARRMoT_InputData_US_v212_up');

dirs = struct('q', fullfile(out_root, 'Q'), 'par', fullfile(out_root, 'Parameters'), ...
              'log', fullfile(out_root, 'calib_log'), 'fail', fullfile(out_root, 'failed'), ...
              'cmaes', fullfile(out_root, 'cmaes_output'));
fn = fieldnames(dirs);
for k = 1:numel(fn)
    if ~exist(dirs.(fn{k}), 'dir'); mkdir(dirs.(fn{k})); end
end

basin_ids = strtrim(splitlines(fileread(basin_list_file)));
basin_ids = basin_ids(~cellfun(@isempty, basin_ids));

flood_tags = {};
for peak = [1, 3, 5]
    for rate = [0, 10, 30, 50]
        flood_tags{end+1} = sprintf('peak%d_rate%d', peak, rate); %#ok<AGROW>
    end
end
flood_tags{end+1} = 'nostorm';

for idx = start_num:min(end_num, numel(basin_ids))
    basin_id = basin_ids{idx};
    q_file = fullfile(dirs.q, sprintf('varssim_%s_Q.csv', basin_id));
    if exist(q_file, 'file') == 2
        fprintf('Output exists for basin %s, skipping.\n', basin_id);
        continue;
    end
    fprintf('\n--- Processing basin: %s ---\n', basin_id);
    t0 = tic;
    try
        original_mat_file = fullfile(input_dir, sprintf('varssim_%s.mat', basin_id));
        [par_opt, cal_info, dates_og] = calibrate_model(original_mat_file, expected_data_start, ...
            cal_start_date, cal_end_date, optimiser_seed, max_iter, dirs.cmaes, basin_id);

        tags = [{'og', 'no_precip'}, flood_tags];
        files = [{original_mat_file, fullfile(no_precip_input_dir, sprintf('varssim_%s_no_precip.mat', basin_id))}, ...
                 cellfun(@(t) fullfile(up_input_dir, sprintf('varssim_%s_up_%s.mat', basin_id, t)), flood_tags, 'UniformOutput', false)];
        Q = NaN(numel(dates_og), numel(tags));
        for k = 1:numel(tags)
            if ~exist(files{k}, 'file')
                error('missing scenario input %s', files{k});
            end
            [q_sim, d] = simulate(files{k}, par_opt, expected_data_start);
            if any(~isfinite(q_sim))
                error('non-finite simulated discharge in scenario %s', tags{k});
            end
            Q(1:numel(q_sim), k) = q_sim(:);
            if ~isequal(d, dates_og(1:numel(d)))
                error('scenario %s dates do not match the original file', tags{k});
            end
        end

        % parameters
        par_tbl = array2table(par_opt(:)', 'VariableNames', strcat('Param_', string(1:numel(par_opt))));
        par_tbl = [table("m37", 'VariableNames', {'Model_Name'}), par_tbl];
        writetable(par_tbl, fullfile(dirs.par, sprintf('varssim_%s_calibrated_parameters.csv', basin_id)));

        % calibration log
        cal_info.runtime_s = toc(t0);
        writetable(struct2table(cal_info), fullfile(dirs.log, sprintf('varssim_%s_calib.csv', basin_id)));

        % discharge (written last: its existence marks the basin complete)
        q_tbl = array2table(Q, 'VariableNames', tags);
        q_tbl = [table(cellstr(datestr(dates_og, 'yyyy-mm-dd')), 'VariableNames', {'Date'}), q_tbl];
        writetable(q_tbl, q_file);
        fprintf('--- Finished basin %s in %.0f s (KGE_cal %.3f, %d evals) ---\n', basin_id, cal_info.runtime_s, cal_info.KGE_cal, cal_info.n_evals);
    catch ME
        msg = getReport(ME, 'extended', 'hyperlinks', 'off');
        fprintf(2, '\n*** FAILED basin %s: %s ***\n', basin_id, msg);
        fid = fopen(fullfile(dirs.fail, sprintf('varssim_%s_failed.txt', basin_id)), 'w');
        fprintf(fid, '%s\n', ME.message);
        fclose(fid);
    end
end
end

% =========================================================================
function m = make_model(climate)
m = feval('m_37_hbv_15p_5s');
m.input_climate = climate;
m.solver_opts.resnorm_tolerance = 0.1;
m.solver_opts.resnorm_maxiter   = 6;
m.S0 = zeros(m.numStores, 1);
end

% =========================================================================
function [par_opt, info, dates] = calibrate_model(data_file_path, expected_data_start, cal_start_date, cal_end_date, seed, max_iter, cmaes_output_dir, basin_id)
[climate, Q_obs, dates] = hbv_load_input(data_file_path, expected_data_start);
m = make_model(climate);

cal_idx = hbv_cal_idx(dates, cal_start_date, cal_end_date);
n_obs_cal = sum(isfinite(Q_obs(cal_idx)));
if n_obs_cal < 365
    error('only %d observed discharge days in the calibration window', n_obs_cal);
end

parRanges = m.parRanges;
optim_opts.insigma    = .3*(parRanges(:,2) - parRanges(:,1));
optim_opts.LBounds    = parRanges(:,1);
optim_opts.UBounds    = parRanges(:,2);
optim_opts.PopSize    = 4 + floor(3*log(m.numParams));
optim_opts.TolX       = 1e-3;
optim_opts.TolFun     = 1e-2;
optim_opts.TolHistFun = 1e-2;
optim_opts.MaxIter    = max_iter;
optim_opts.Seed       = seed;
run_identifier = sprintf('sim%s_model01', basin_id);
optim_opts.SaveFilename      = fullfile(cmaes_output_dir, ['cmaesvars_' run_identifier '.mat']);
optim_opts.LogFilenamePrefix = fullfile(cmaes_output_dir, ['cmaes_log_' run_identifier '_']);
optim_opts.EvalParallel = false;
optim_opts.DispModulo   = -1;

par_ini = mean(parRanges, 2);   % always the centre of the ranges: no warm start from earlier runs
[par_opt, of_cal, stopflag, output] = m.calibrate(Q_obs, cal_idx, 'my_cmaes', par_ini, optim_opts, 'of_KGE', 1, 1, [1,1,1]);

info.basin       = string(basin_id);
info.data_start  = string(datestr(dates(1), 'yyyy-mm-dd'));
info.cal_start   = string(datestr(cal_start_date, 'yyyy-mm-dd'));
info.cal_end     = string(datestr(cal_end_date, 'yyyy-mm-dd'));
info.n_obs_cal   = n_obs_cal;
info.objective   = "of_KGE";
info.KGE_cal     = of_cal;
info.optimiser   = "my_cmaes";
info.n_evals     = output.evals;
info.stopflag    = string(strjoin(cellfun(@char, stopflag, 'UniformOutput', false), '|'));
info.PopSize     = optim_opts.PopSize;
info.MaxIter     = optim_opts.MaxIter;
info.max_evals   = optim_opts.PopSize * optim_opts.MaxIter;
info.TolX        = optim_opts.TolX;
info.TolFun      = optim_opts.TolFun;
info.TolHistFun  = optim_opts.TolHistFun;
info.seed        = seed;
end

% =========================================================================
function [Q_sim, dates] = simulate(data_file_path, par_opt, expected_data_start)
[climate, ~, dates] = hbv_load_input(data_file_path, expected_data_start);
m = make_model(climate);
Q_sim = m.get_streamflow([], [], par_opt);
end
