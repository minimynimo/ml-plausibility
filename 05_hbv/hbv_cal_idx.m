function cal_idx = hbv_cal_idx(dates, cal_start_date, cal_end_date)
% Indices of the calibration window in a simulation that starts on dates(1). Errors unless the
% window is fully covered, so a wrong start date can never silently shift the calibration period.
cal_idx = find(dates >= cal_start_date & dates <= cal_end_date);
if isempty(cal_idx) || dates(cal_idx(1)) ~= cal_start_date || dates(cal_idx(end)) ~= cal_end_date ...
        || numel(cal_idx) ~= days(cal_end_date - cal_start_date) + 1
    error('calibration window %s..%s not covered', datestr(cal_start_date), datestr(cal_end_date));
end
end
