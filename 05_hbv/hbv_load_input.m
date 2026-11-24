function [climate, Q_obs, dates] = hbv_load_input(data_file_path, expected_data_start)
S = load(data_file_path);
d = S.data_MARRMoT_examples;
dates = datetime(d.dates_as_datenum(:), 'ConvertFrom', 'datenum');
if dates(1) ~= expected_data_start
    error('%s starts on %s, expected %s', data_file_path, datestr(dates(1)), datestr(expected_data_start));
end
if any(days(diff(dates)) ~= 1)
    error('%s: dates are not consecutive days', data_file_path);
end
climate.precip  = d.precipitation;
climate.temp    = d.temperature;
climate.pet     = d.potential_evapotranspiration;
climate.delta_t = 1;
if any(~isfinite([climate.precip; climate.temp; climate.pet]))
    error('%s: non-finite forcing', data_file_path);
end
Q_obs = d.streamflow;
end

