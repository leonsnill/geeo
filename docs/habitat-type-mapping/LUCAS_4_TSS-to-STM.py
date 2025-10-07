'''
Mapping pan-European Land Cover - Part 4
This file creates STMs from the TSS .csv data.
'''
import numpy as np
from tqdm import tqdm
import pandas as pd

df_lucas = pd.read_csv('TSS_LUCAS_HARMO_V1_EO_LC_EU.csv')

'''
Spectral-Temporal-Metrics (STMs)

- LUCAS data for 2015 and 2018, use only data from 2014-2016 / 2017-2019 either for each plot ID
- EUNIS: use data for 2017-2019

Temporal windows
- Full year
- winter (Dec, Jan, Feb)
- spring (Mar, Apr, May)
- summer (Jun, Jul, Aug)
- autumn (Sep, Oct, Nov)

Metrics
- p5
- p25
- p50
- p75
- p95
- stdDev

Output format, for example:
winter_NDVI_p5
'''

# ---------------------------------------------------------------------------------------------------
# inpit data
group_col_lucas = 'id'
value_cols = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2', 'MDWI', 'NBR', 'NDVI']
nodata = -9999

def prep(df: pd.DataFrame, date_col='YYYYMMDD') -> pd.DataFrame:
    existing = [c for c in value_cols if c in df.columns]
    df[existing] = df[existing].replace(nodata, np.nan)
    dt = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
    df = df.loc[dt.notna()].copy()
    df.index = dt[dt.notna()]
    df.sort_index(inplace=True)
    return df

df_lucas = prep(df_lucas, 'YYYYMMDD')

SEASONS = {
    'full': list(range(1, 13)),
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'autumn': [9, 10, 11],
}
PCTS = [5, 25, 50, 75, 95]

# ---------------------------------------------------------------------------------------------------
# funsctions
def _compute_metrics_for_frame(frame: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    if frame.empty:
        for col in cols:
            for p in PCTS:
                out[f'{col}_p{p}'] = np.nan
            out[f'{col}_stdDev'] = np.nan
        return out

    for col in cols:
        arr = frame[col].to_numpy(dtype=float)
        # All-NaN safe
        if arr.size == 0 or np.isnan(arr).all():
            for p in PCTS:
                out[f'{col}_p{p}'] = np.nan
            out[f'{col}_stdDev'] = np.nan
            continue
        for p in PCTS:
            out[f'{col}_p{p}'] = np.nanpercentile(arr, p)
        out[f'{col}_stdDev'] = np.nanstd(arr)
    return out


def _stm_for_group(df_grp: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cols: list[str]) -> dict:
    # Restrict to the 3-year window
    ts = df_grp.loc[(df_grp.index >= start) & (df_grp.index <= end)]
    result = {}
    for season_name, months in SEASONS.items():
        if season_name == 'full':
            sub = ts
        else:
            sub = ts[ts.index.month.isin(months)]
        metrics = _compute_metrics_for_frame(sub[cols], cols)
        # Prefix with season
        for k, v in metrics.items():
            result[f'{season_name}_{k}'] = v
    return result


def _lucas_date_range_for_group(df_grp: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if 'year' not in df_grp.columns:
        raise ValueError("LUCAS dataframe must contain a 'year' column per point.")
    years = df_grp['year'].dropna()
    if years.empty:
        raise ValueError("No survey year found for a LUCAS point.")
    # use mode; if tie, pick the max
    survey_year = years.mode().iloc[0]
    try:
        survey_year = int(survey_year)
    except Exception:
        raise ValueError(f"Invalid survey year value: {survey_year}")

    if survey_year == 2015:
        return (pd.Timestamp('2014-01-01'), pd.Timestamp('2016-12-31'))
    elif survey_year == 2018:
        return (pd.Timestamp('2017-01-01'), pd.Timestamp('2019-12-31'))
    else:
        raise ValueError(f"Unsupported LUCAS survey year {survey_year}; expected 2015 or 2018.")


def compute_stms(
    df: pd.DataFrame,
    group_col: str,
    cols: list[str],
    date_range_getter
) -> pd.DataFrame:
    # filter to available feature columns
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("None of the requested feature columns are present in the dataframe.")

    rows = []
    ids = []
    # group by point id/fid
    for gid, grp in tqdm(df.groupby(group_col, sort=False)):
        start, end = date_range_getter(grp)
        metrics = _stm_for_group(grp, start, end, cols)
        rows.append(metrics)
        ids.append(gid)

    out = pd.DataFrame(rows, index=ids)
    out.index.name = group_col
    return out

# ---------------------------------------------------------------------------------------------------
# compute STMs
df_lucas_stm = compute_stms(
    df_lucas,
    group_col_lucas,
    value_cols,
    date_range_getter=_lucas_date_range_for_group
)

# re-join based onn id
attr_lucas= ['LC_ID', 'LC_NAME', 'lc1', 'lc1_label', 'lc1_perc', 'parcel_area_ha', 'revisit', 'year', '.geo']

# re-join based on id
df_lucas_stm = df_lucas_stm.merge(df_lucas[attr_lucas + [group_col_lucas]].drop_duplicates(), on=group_col_lucas, how='left')

# write both to disc
df_lucas_stm.to_csv('STM_LUCAS_HARMO_V1_EO_LC_EU.csv')

# ---------------------------------------------------------------------------------------------------
# merge with CHELSA + DEM data
df_lucas_stm = pd.read_csv('STM_LUCAS_HARMO_V1_EO_LC_EU.csv')
df_lucas_dem = pd.read_csv('LUCAS_HARMO_V1_EO_LC_CHELSA_DEM.csv')
col_to_join = ['DEM', 'slope', 'northness', 'eastness'] + [f'bio{i}' for i in range(1, 20)]
df_lucas_stm = df_lucas_stm.merge(df_lucas_dem[['id'] + col_to_join], on='id', how='left')

cols_to_keep = [
    'id', 'LC_ID', 'year', '.geo',
    'DEM', 'slope', 'northness', 'eastness',
    'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'bio10',
    'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19',
]

prefixes = ['full_', 'summer_', 'spring_', 'autumn_', 'winter_']
cols_to_keep += [col for col in df_lucas_stm.columns if any(col.startswith(p) for p in prefixes)]

df_lucas_stm = df_lucas_stm[cols_to_keep]
df_lucas_stm = df_lucas_stm.dropna(subset=['LC_ID'])
df_lucas_stm.loc[:, 'LC_ID'] = df_lucas_stm['LC_ID'].astype(int)

# write to disc
df_lucas_stm.to_csv('STM_LUCAS_HARMO_V1_EO_LC_EU_final.csv', index=False)

# EOF