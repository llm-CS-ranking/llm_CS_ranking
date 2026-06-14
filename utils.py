import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken


DATE_KEYWORDS = (
    "date",
    "period end",
    "period-end",
    "balance sheet date",
    "as reported period end",
)

_DATE_RE = re.compile(r"^\d{8}$")

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens for a given text using tiktoken's model-specific encoding.
    """
    # Pick encoding based on model; fallback to a reasonable default
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")  # good modern default

    return len(enc.encode(text))

def _normalize_end_date(end_date_or_month: str) -> pd.Timestamp:
    s = str(end_date_or_month)
    if len(s) == 7 and s[4] == "-":  # 'YYYY-MM'
        return (pd.to_datetime(s + "-01") + pd.offsets.MonthEnd(0)).normalize()
    return pd.to_datetime(s).normalize()

def monthly_last_quarter_with_last4q(
    df: pd.DataFrame,
    value_col: str = "DATAITEMVALUE",
    filing_col: str = "FILINGDATE",
    item_col: str = "DATAITEMNAME",
    quarter_level: str | int = "QUARTER",
    company_level: str | int = "COMPANYID",
    fill_value: float = 0.0,
    asof: str | None = None,   # cutoff 끝을 여기까지 생성(월 라벨 생성용)
) -> pd.DataFrame:
    """
    MultiIndex(QUARTER, COMPANYID) + columns=[FILINGDATE, DATAITEMNAME, VALUE...] 에서

    ✅ 각 MONTH='YYYY-MM'의 결과는:
       filing_date < 해당 월의 1일 00:00:00 (포함 안 함)
       인 행만 사용해서 회사+아이템별 최신분기 기준 최근 4개 분기를 wide로 생성.

    Output columns:
      MONTH, COMPANYID, DATAITEMNAME, VALUE_Q1..VALUE_Q4
      (Q4 최신, Q1 가장 오래된)
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df.index must be a MultiIndex with QUARTER and COMPANYID levels.")

    work = df.reset_index()

    idx_names = list(df.index.names)
    q_col = idx_names[quarter_level] if isinstance(quarter_level, int) else quarter_level
    c_col = idx_names[company_level] if isinstance(company_level, int) else company_level

    for col in (filing_col, item_col, value_col):
        if col not in work.columns:
            raise ValueError(f"Need column: {col}")

    work[filing_col] = pd.to_datetime(work[filing_col], errors="coerce")
    work = work.dropna(subset=[filing_col])

    # QUARTER 정렬 가능화
    if not isinstance(work[q_col].dtype, pd.PeriodDtype):
        try:
            work[q_col] = pd.PeriodIndex(work[q_col].astype(str), freq="Q")
        except Exception:
            q_dt = pd.to_datetime(work[q_col], errors="coerce")
            if q_dt.notna().any():
                work[q_col] = q_dt

    # (회사, 아이템, 분기)별 마지막 filing만
    work = work.sort_values([c_col, item_col, q_col, filing_col])
    work = work.drop_duplicates(subset=[c_col, item_col, q_col], keep="last")

    # ✅ 월 라벨 생성 범위: (데이터 min filing month) ~ (max filing month 또는 asof month)
    min_d = work[filing_col].min()
    max_d_data = work[filing_col].max()

    if asof is not None:
        asof_dt = pd.to_datetime(asof)
        max_d = max(max_d_data, asof_dt)
    else:
        max_d = max_d_data

    # ✅ 각 월 라벨(YYYY-MM)과 cutoff = 그 달 1일 00:00 생성
    month_starts = pd.date_range(
        pd.Period(min_d, "M").start_time,
        pd.Period(max_d, "M").start_time,
        freq="MS",
    )
    month_labels = month_starts.to_period("M").astype(str)
    cutoffs = month_starts  # ✅ cutoff = 해당 월 시작
    # 필터는 filing_date < cutoff (포함 안 함)

    out_frames = []
    for cutoff, mlabel in zip(cutoffs, month_labels):
        # ✅ 핵심: 해당 월 1일 이전(포함 안함)
        w = work.loc[work[filing_col] < cutoff]
        if w.empty:
            continue

        w = (w.sort_values([c_col, item_col, q_col])
               .groupby([c_col, item_col], as_index=False)
               .tail(4))
        w = w.sort_values([c_col, item_col, q_col])
        w["q_idx"] = w.groupby([c_col, item_col]).cumcount() + 1  # 1..4

        wide = w.pivot(index=[c_col, item_col], columns="q_idx", values=value_col)
        wide = wide.reindex(columns=[1, 2, 3, 4])
        wide.columns = [f"{value_col}_Q{i}" for i in range(1, 5)]

        wide = wide.reset_index()
        wide.insert(0, "MONTH", mlabel)
        wide = wide.fillna(fill_value)
        out_frames.append(wide)

    if not out_frames:
        return pd.DataFrame(columns=[
            "MONTH", c_col, item_col,
            f"{value_col}_Q1", f"{value_col}_Q2", f"{value_col}_Q3", f"{value_col}_Q4"
        ])

    return pd.concat(out_frames, ignore_index=True)

def _to_yyyy_mm_dd_series(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    s = s.round().astype("Int64")
    s = s.astype(str).replace("<NA>", pd.NA)
    s = s.str.replace(r"\D", "", regex=True)

    mask = s.str.match(_DATE_RE, na=False)
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out.loc[mask] = s.loc[mask].str.slice(0, 4) + "-" + s.loc[mask].str.slice(4, 6) + "-" + s.loc[mask].str.slice(6, 8)
    return out

def build_llm_payload(
    df: pd.DataFrame,
    month: str,
    top_k_items: int | None = 40,
    decimals: int = 2,
    fill_value: float = 0.0,
    item_col: str = "DATAITEMNAME",
    q_cols: tuple[str, str, str, str] = (
        "DATAITEMVALUE_Q1",
        "DATAITEMVALUE_Q2",
        "DATAITEMVALUE_Q3",
        "DATAITEMVALUE_Q4",
    ),
    month_level: str = "MONTH",
    ticker_level: str = "TICKERSYMBOL",
    return_json_string: bool = False,
):
    # 1) 월 슬라이스
    mdf = df.xs(month, level=month_level).copy()
    mdf = mdf[[item_col, *q_cols]].fillna(fill_value).reset_index()

    # 2) ticker 컬럼 확보
    if ticker_level not in mdf.columns:
        mdf.rename(columns={mdf.columns[0]: ticker_level}, inplace=True)

    # 3) 회사별 top-k item 선택
    if top_k_items is not None:
        mdf["_score"] = mdf[q_cols[3]].abs()
        mdf = (
            mdf.sort_values([ticker_level, "_score"], ascending=[True, False])
               .groupby(ticker_level, as_index=False)
               .head(top_k_items)
               .drop(columns="_score")
        )

    # 4) 날짜 항목 판별
    name_l = mdf[item_col].astype(str).str.lower()
    date_mask = np.zeros(len(mdf), dtype=bool)
    for k in DATE_KEYWORDS:
        date_mask |= name_l.str.contains(k, na=False)

    # 5) Q1~Q4 값 정리
    for qc in q_cols:
        # 혼합 dtype 허용
        mdf[qc] = mdf[qc].astype("object")

        # 날짜 → 문자열
        date_str = _to_yyyy_mm_dd_series(mdf.loc[date_mask, qc])
        mdf.loc[date_mask, qc] = date_str.astype(object)

        # 숫자 처리
        non = ~date_mask
        v = pd.to_numeric(mdf.loc[non, qc], errors="coerce").fillna(fill_value)

        is_int = (v - np.round(v)).abs() < 1e-9
        mdf.loc[non & is_int, qc] = np.round(v[is_int]).astype("int64").astype(object)
        mdf.loc[non & ~is_int, qc] = v[~is_int].round(decimals).astype(float).astype(object)

    # 6) 컬럼 이름 정리
    rec = mdf.rename(columns={
        item_col: "name",
        q_cols[0]: "Q1",
        q_cols[1]: "Q2",
        q_cols[2]: "Q3",
        q_cols[3]: "Q4",
    })

    # 7) payload 생성
    payload = []
    for ticker, g in rec.groupby(ticker_level, sort=True):
        payload.append({
            "MONTH": month,
            "TICKER": str(ticker),
            "items": g[["name", "Q1", "Q2", "Q3", "Q4"]].to_dict(orient="records"),
        })

    if return_json_string:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return payload

def _r(x, decimals: int):
    """None/NaN/inf 안전 반올림"""
    try:
        if x is None or np.isnan(x) or np.isinf(x):
            return 0.0
    except Exception:
        pass
    return float(np.round(x, decimals))


def build_market_summary(
    df: pd.DataFrame,
    end_date_or_month: str,
    lookback_days: int = 63,
    price_col: str = "DIV_ADJ_CLOSE",
    vol_col: str = "VOLUME",
    date_col: str = "DATE",
    ticker_col: str = "TICKERSYMBOL",
    decimals: int = 4,          # ✅ 비율/지표 반올림 자릿수
    volume_decimals: int = 0,   # ✅ 평균 거래량 반올림(보통 정수)
):
    """
    - reset_index 기반 (DATE dtype 섞임/unsorted 회피)
    - end_date_or_month: 'YYYY-MM' 또는 'YYYY-MM-DD'
    """
    end = _normalize_end_date(end_date_or_month)
    start = end - pd.Timedelta(days=int(lookback_days * 1.5))  # 휴일 버퍼

    mdf = df.reset_index()
    mdf[date_col] = pd.to_datetime(mdf[date_col], errors="coerce")
    mdf = mdf.dropna(subset=[date_col])

    mdf = mdf[(mdf[date_col] >= start) & (mdf[date_col] <= end)]
    mdf = mdf.sort_values([ticker_col, date_col])

    out = []
    for ticker, g in mdf.groupby(ticker_col):
        g = g.tail(lookback_days)
        if len(g) < 20:
            continue

        close = pd.to_numeric(g[price_col], errors="coerce").ffill().bfill()
        ret = close.pct_change()

        vol_s = pd.to_numeric(g[vol_col], errors="coerce").fillna(0)

        # 미리 계산(중복 연산 제거)
        ma20 = close.tail(20).mean()
        ma60 = close.mean()
        last = close.iloc[-1]
        hi = close.max()

        avg_vol = vol_s.mean()
        std_vol = vol_s.std() + 1e-9
        vol_z = (vol_s.iloc[-1] - avg_vol) / std_vol

        out.append({
            "TICKER": str(ticker),
            "asof": end.strftime("%Y-%m-%d"),

            # returns
            "ret_1m": _r(last / close.iloc[-21] - 1, decimals) if len(close) >= 21 else 0.0,
            "ret_3m": _r(last / close.iloc[0] - 1, decimals),

            # realized vol (annualized)
            "vol_1m": _r(ret.tail(21).std() * np.sqrt(252), decimals),
            "vol_3m": _r(ret.std() * np.sqrt(252), decimals),

            # trend gaps
            "ma20_gap": _r(last / ma20 - 1, decimals),
            "ma60_gap": _r(last / ma60 - 1, decimals),

            # volume (크면 정수로)
            "avg_volume": _r(avg_vol, volume_decimals),
            "volume_z": _r(vol_z, decimals),

            # drawdown
            "drawdown_from_high": _r(last / hi - 1, decimals),
        })

    return out

def llm_json_to_df(result) -> pd.DataFrame:
    """
    OpenAI chat.completions 결과에서 JSON을 뽑아 DataFrame으로 변환.
    - content가 JSON만 있어도 OK
    - content에 텍스트가 섞여 있어도 JSON 블록을 찾아 파싱 시도
    """
    content = result
    if content is None:
        raise ValueError("LLM response content is None")

    s = content.strip()

    # 1) 먼저 그대로 파싱 시도
    try:
        data = json.loads(s)
        return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass

    # 2) 코드펜스 ```json ... ``` 제거 시도
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(s2)
        return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass

    # 3) 문자열 안에서 첫 번째 유효 JSON 배열/객체만 파싱
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"[\[\{]", s)]:
        try:
            data, end = decoder.raw_decode(s[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            rest = s[start + end :].lstrip()
            if rest.startswith(","):
                wrapped_from_start = "[" + s[start:].strip().strip(",") + "]"
                try:
                    data = json.loads(wrapped_from_start)
                    return pd.DataFrame(data)
                except json.JSONDecodeError:
                    pass
            return pd.DataFrame([data])

    # 4) 일부 모델은 {"ticker":...},{"ticker":...}처럼 배열 괄호 없이 반환
    wrapped = "[" + s2.strip().strip(",") + "]"
    try:
        data = json.loads(wrapped)
        return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not find JSON in model output:\n{s[:500]}")

def monthly_rank_panel(
    monthly_rank_dfs: dict[str, pd.DataFrame] | list[tuple[str, pd.DataFrame]],
    month_col: str = "MONTH",
    ticker_col: str = "ticker",
    rank_col: str = "rank",
    sort_month: bool = True,
    dtype: str | None = "Int64",
) -> pd.DataFrame:
    """
    월별 rank 결과들을 모아:
      - index: YYYY-MM (월)
      - columns: 모든 ticker
      - values: rank

    Parameters
    ----------
    monthly_rank_dfs:
        1) dict 형태: {"2025-11": df, "2025-12": df, ...}
        2) list 형태: [("2025-11", df), ("2025-12", df), ...]
        df는 최소한 [ticker, rank] 컬럼을 가져야 함.
        (month가 df 안에 들어있다면 list/dict 대신 단일 df로 처리하는 함수도 아래에 제공)

    Returns
    -------
    pd.DataFrame:
        panel_df.loc["2025-11","AAPL"] == 2 같은 형태.
        rank가 없는 ticker/month는 NaN (또는 dtype Int64면 <NA>)
    """
    if isinstance(monthly_rank_dfs, dict):
        items = list(monthly_rank_dfs.items())
    else:
        items = list(monthly_rank_dfs)

    frames = []
    for m, df in items:
        tmp = df[[ticker_col, rank_col]].copy()
        tmp[month_col] = m
        frames.append(tmp)

    long_df = pd.concat(frames, ignore_index=True)

    panel = (
        long_df.pivot_table(
            index=month_col,
            columns=ticker_col,
            values=rank_col,
            aggfunc="min",   # 중복 있으면 최소값 사용(보통 중복 없음)
        )
    )

    if sort_month:
        # YYYY-MM 문자열 정렬이 시간순과 동일하지만 안전하게 Period로 정렬
        panel = panel.sort_index(key=lambda x: pd.PeriodIndex(x, freq="M"))

    if dtype is not None:
        panel = panel.astype(dtype)

    return panel

def backtest_quintiles_with_universe_ew_capw(
    rank_panel: pd.DataFrame,          # index=MONTH("YYYY-MM"), columns=ticker, values=rank (작을수록 좋음)
    market_df: pd.DataFrame,           # MultiIndex(DATE,TICKERSYMBOL) or columns DATE,TICKERSYMBOL
    price_col: str = "DIV_ADJ_CLOSE",
    cap_col: str = "MKTCAP",
    n_quantiles: int = 5,
    lag_days: int = 1,                 # trading lag: 월초 첫 거래일 + lag_days에 리밸런싱
    cost_bps: float = 10.0,            # turnover * cost_bps/10000 을 리밸런싱일에 차감
    min_names_per_bucket: int = 1,
    add_long_short: bool = True,
) -> pd.DataFrame:
    """
    Returns
    -------
    daily_returns: pd.DataFrame
      columns:
        Q1..Q{n_quantiles}  (각 분위 Equal Weight)
        UNIV_EW             (유니버스 Equal Weight)
        UNIV_CAPW           (유니버스 Cap Weight, MKTCAP)
        (optional) Q1_minus_Q{n_quantiles}
    """

    # ---------------------------
    # 0) market_df -> wide prices, returns, cap_wide
    # ---------------------------
    if isinstance(market_df.index, pd.MultiIndex):
        m = market_df.reset_index()
    else:
        m = market_df.copy()

    need_cols = {"DATE", "TICKERSYMBOL", price_col, cap_col}
    missing = need_cols - set(m.columns)
    if missing:
        raise ValueError(f"market_df missing columns: {missing}")

    m["DATE"] = pd.to_datetime(m["DATE"])
    m = m.sort_values(["DATE", "TICKERSYMBOL"])

    prices = m.pivot(index="DATE", columns="TICKERSYMBOL", values=price_col).sort_index()
    caps   = m.pivot(index="DATE", columns="TICKERSYMBOL", values=cap_col).sort_index()

    # 일별 수익률 (중간 NA를 자동 fill 하지 않음)
    rets = prices.pct_change(fill_method=None)
    rets0 = rets.fillna(0.0)

    all_dates = prices.index
    tickers = prices.columns

    # rank_panel month index 정리
    rp = rank_panel.copy()
    rp.index = rp.index.astype(str)

    # ---------------------------
    # 1) 월별 실행일(exec_date) 스케줄: 해당 월 1일 이후 첫 거래일 + lag_days
    # ---------------------------
    sched = []
    for ms in rp.index:
        month_start = pd.Period(ms, freq="M").start_time  # YYYY-MM-01 00:00:00
        pos0 = all_dates.searchsorted(month_start, side="left")
        if pos0 >= len(all_dates):
            continue
        pos_exec = pos0 + lag_days
        if pos_exec >= len(all_dates):
            continue
        sched.append((ms, all_dates[pos_exec]))

    if len(sched) < 2:
        raise ValueError("Not enough months within market date range to backtest.")

    sched = (
        pd.DataFrame(sched, columns=["MONTH", "EXEC_DATE"])
        .drop_duplicates("EXEC_DATE", keep="last")
        .sort_values("EXEC_DATE")
        .reset_index(drop=True)
    )

    # ---------------------------
    # 2) 유틸: 월별 rank -> 5분위 bucket (Q1이 best)
    # ---------------------------
    def _month_to_buckets(rank_row: pd.Series) -> list[list[str]]:
        s = rank_row.dropna()
        if s.empty:
            return [[] for _ in range(n_quantiles)]
        s = s.sort_values(kind="mergesort")  # rank 낮을수록 좋음
        names = [t for t in s.index.to_list() if t in tickers]
        return [list(arr) for arr in np.array_split(names, n_quantiles)]

    month_buckets = {ms: _month_to_buckets(rp.loc[ms]) for ms in sched["MONTH"].unique() if ms in rp.index}

    # ---------------------------
    # 3) 포트 세팅: Q1..Qn (EW), UNIV_EW, UNIV_CAPW
    # ---------------------------
    port_names = [f"Q{i+1}" for i in range(n_quantiles)] + ["UNIV_EW", "UNIV_CAPW"]

    W = {p: pd.DataFrame(0.0, index=all_dates, columns=tickers) for p in port_names}
    prev_w = {p: pd.Series(0.0, index=tickers) for p in port_names}
    costs = {p: pd.Series(0.0, index=all_dates) for p in port_names}

    def _cap_weights_on_date(d: pd.Timestamp, names: list[str]) -> pd.Series:
        """
        d(리밸런싱일) 기준, caps에서 d 이전/포함 마지막 값을 가져와 cap-weight 산출.
        """
        out = pd.Series(0.0, index=tickers)
        if len(names) == 0:
            return out

        pos = caps.index.searchsorted(d, side="right") - 1
        if pos < 0:
            return out

        cap_row = caps.iloc[pos].reindex(names)
        cap_row = pd.to_numeric(cap_row, errors="coerce").fillna(0.0)
        s = cap_row.sum()
        if s > 0:
            out.loc[names] = (cap_row / s).values
        return out

    # ---------------------------
    # 4) 리밸런싱 루프
    # ---------------------------
    for i in range(len(sched)):
        ms = sched.loc[i, "MONTH"]
        d0 = sched.loc[i, "EXEC_DATE"]
        d1 = sched.loc[i + 1, "EXEC_DATE"] if i + 1 < len(sched) else (all_dates[-1] + pd.Timedelta(days=1))
        mask = (all_dates >= d0) & (all_dates < d1)

        # 월별 유니버스: 그 달 rank가 부여된 종목
        if ms in rp.index:
            univ = rp.loc[ms].dropna().index
            univ = [t for t in univ.to_list() if t in tickers]
        else:
            univ = []

        buckets = month_buckets.get(ms, [[] for _ in range(n_quantiles)])

        # ---- Q1..Qn: 분위별 Equal Weight ----
        for q in range(n_quantiles):
            pname = f"Q{q+1}"
            names = buckets[q]
            new_w = pd.Series(0.0, index=tickers)
            if len(names) >= min_names_per_bucket:
                new_w.loc[names] = 1.0 / len(names)

            turnover = float((new_w - prev_w[pname]).abs().sum())
            costs[pname].loc[d0] = turnover * (cost_bps / 10000.0)

            W[pname].loc[mask, :] = new_w.values
            prev_w[pname] = new_w

        # ---- UNIV_EW ----
        new_uew = pd.Series(0.0, index=tickers)
        if len(univ) >= min_names_per_bucket:
            new_uew.loc[univ] = 1.0 / len(univ)

        turnover = float((new_uew - prev_w["UNIV_EW"]).abs().sum())
        costs["UNIV_EW"].loc[d0] = turnover * (cost_bps / 10000.0)

        W["UNIV_EW"].loc[mask, :] = new_uew.values
        prev_w["UNIV_EW"] = new_uew

        # ---- UNIV_CAPW ----
        new_ucw = _cap_weights_on_date(d0, univ)

        turnover = float((new_ucw - prev_w["UNIV_CAPW"]).abs().sum())
        costs["UNIV_CAPW"].loc[d0] = turnover * (cost_bps / 10000.0)

        W["UNIV_CAPW"].loc[mask, :] = new_ucw.values
        prev_w["UNIV_CAPW"] = new_ucw

    # ---------------------------
    # 5) 일별 포트 수익률: 전일 비중 적용 + 리밸런싱일 비용 차감
    # ---------------------------
    daily = {}
    for pname in port_names:
        daily[pname] = (W[pname].shift(1).fillna(0.0) * rets0).sum(axis=1) - costs[pname]

    daily_df = pd.DataFrame(daily).sort_index()

    if add_long_short and n_quantiles >= 2:
        long_name = "Q1"
        short_name = f"Q{n_quantiles}"
        long_gross = (W[long_name].shift(1).fillna(0.0) * rets0).sum(axis=1)
        short_gross = (W[short_name].shift(1).fillna(0.0) * rets0).sum(axis=1)
        daily_df["Q1_minus_Q5"] = (
            long_gross
            - short_gross
            - costs[long_name]
            - costs[short_name]
        )

    first_exec = sched["EXEC_DATE"].iloc[0]
    daily_df = daily_df.loc[first_exec:]

    return daily_df

def plot_cumulative_returns(
    daily_returns: pd.DataFrame | pd.Series,
    title: str = "Cumulative Returns",
    start_value: float = 1.0,
    logy: bool = False,
    dropna: bool = False,
):
    """
    일별 수익률(Series 또는 DataFrame) -> 누적수익 곡선 그리기.

    Parameters
    ----------
    daily_returns : pd.Series or pd.DataFrame
        index는 DatetimeIndex 권장, 값은 일별 수익률(예: 0.01 = +1%)
    start_value : float
        누적 시작값(보통 1.0)
    logy : bool
        y축 로그 스케일
    dropna : bool
        True면 NaN이 있는 구간을 제거, False면 NaN을 0으로 간주

    Returns
    -------
    equity : pd.Series or pd.DataFrame
        누적수익(= start_value * cumprod(1+r))
    """
    r = daily_returns.copy()

    # 정렬/타입 정리
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)
    r = r.sort_index()

    if dropna:
        r = r.dropna()
    else:
        r = r.fillna(0.0)

    # 누적수익
    equity = start_value * (1.0 + r).cumprod()

    ax = equity.plot(figsize=(12, 5), linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity" if start_value == 1.0 else f"Equity (start={start_value})")
    ax.grid(True, alpha=0.3)
    if logy:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.show()

    return equity

def portfolio_metrics(
    daily_returns: pd.Series | pd.DataFrame,
    annualization: int = 252,
    rf_daily: float = 0.0,          # 일간 무위험수익률(없으면 0)
    mar_daily: float = 0.0,         # Sortino용 최소수익률(MAR), 일간 기준(없으면 0)
) -> pd.DataFrame:
    """
    일별 수익률(Series/DataFrame)로 성과지표 계산.
    - CAGR(대략), Ann.Return, Ann.Vol, Sharpe, Sortino
    - MDD(Max Drawdown), Calmar
    - Skew, Kurtosis
    - Hit ratio(양수 수익률 비율)

    Returns: index=지표, columns=전략(Series면 1컬럼)
    """

    r = daily_returns.copy()
    if isinstance(r, pd.Series):
        r = r.to_frame(r.name or "strategy")

    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)
    r = r.sort_index()

    # NaN은 0으로 간주(백테스트 결과에 NaN이 섞일 때 편의)
    r = r.fillna(0.0)

    # 초과수익
    ex = r - rf_daily

    # 누적/드로우다운
    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mdd = drawdown.min()

    # 연환산 지표
    ann_ret = (1.0 + r.mean()) ** annualization - 1.0
    ann_vol = r.std(ddof=0) * np.sqrt(annualization)

    # Sharpe. Use np.divide(..., where=...) so zero-volatility columns do not
    # trigger RuntimeWarning before being converted to NaN.
    excess_ann = ex.mean().to_numpy() * annualization
    ann_vol_arr = ann_vol.to_numpy()
    sharpe = np.full_like(excess_ann, np.nan, dtype=float)
    np.divide(excess_ann, ann_vol_arr, out=sharpe, where=np.isfinite(ann_vol_arr) & (ann_vol_arr != 0))
    sharpe = pd.Series(sharpe, index=r.columns)

    # Sortino (downside deviation relative to MAR)
    downside = (r - mar_daily).clip(upper=0.0)
    downside_dev = downside.pow(2).mean().pow(0.5) * np.sqrt(annualization)
    downside_dev_arr = downside_dev.to_numpy()
    excess_downside_ann = (r.mean() - mar_daily).to_numpy() * annualization
    sortino = np.full_like(excess_downside_ann, np.nan, dtype=float)
    np.divide(
        excess_downside_ann,
        downside_dev_arr,
        out=sortino,
        where=np.isfinite(downside_dev_arr) & (downside_dev_arr != 0),
    )
    sortino = pd.Series(sortino, index=r.columns)

    # 기간 기반 CAGR (일수 기반)
    n_days = (r.index[-1] - r.index[0]).days
    years = max(n_days / 365.25, 1e-9)
    cagr = equity.iloc[-1].pow(1.0 / years) - 1.0

    # Calmar (CAGR / |MDD|)
    calmar = cagr / (-mdd.replace(0.0, np.nan))

    # 기타
    hit = (r > 0).mean()
    skew = r.skew()
    kurt = r.kurtosis()

    out = pd.DataFrame({
        "CAGR": cagr,
        "Ann.Return(geom)": ann_ret,
        "Ann.Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "Calmar": calmar,
        "HitRatio": hit,
        "Skew": skew,
        "Kurtosis": kurt,
    }).T

    return out


def drawdown_series(daily_returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    누적수익 기반 드로우다운 시계열을 반환.
    """
    r = daily_returns.copy()
    if isinstance(r, pd.Series):
        r = r.to_frame(r.name or "strategy")

    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)
    r = r.sort_index().fillna(0.0)

    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0

    return dd if isinstance(daily_returns, pd.DataFrame) else dd.iloc[:, 0]

def rank_ic_by_month(
    rank_panel: pd.DataFrame,              # index=MONTH("YYYY-MM"), columns=ticker, values=pred_rank (작을수록 좋음)
    market_df: pd.DataFrame,               # MultiIndex(DATE,TICKERSYMBOL) or columns DATE,TICKERSYMBOL
    price_col: str = "DIV_ADJ_CLOSE",
    lag_months: int = 1,                   # "향후 1개월 수익률"이면 1, 같은 달이면 0
    method: str = "spearman",              # "spearman" or "kendall"
    min_n: int = 5,                        # 한 달에 최소 비교 종목 수
) -> pd.DataFrame:
    """
    각 MONTH의 예측 순위(rank_panel) vs (MONTH + lag_months)의 월수익률 순위 간
    순위 상관계수(IC)를 계산.

    Returns
    -------
    DataFrame with columns:
      - N: 비교 종목 수
      - IC: rank correlation
      - month_ret_rank: (옵션 디버깅용) 실제 수익률 순위
    """

    # ---- market -> monthly returns wide ----
    if isinstance(market_df.index, pd.MultiIndex):
        m = market_df.reset_index()
    else:
        m = market_df.copy()

    if "DATE" not in m.columns or "TICKERSYMBOL" not in m.columns:
        raise ValueError("market_df must have MultiIndex(DATE,TICKERSYMBOL) or columns DATE,TICKERSYMBOL")

    m["DATE"] = pd.to_datetime(m["DATE"])
    m = m.sort_values(["DATE", "TICKERSYMBOL"])
    prices = m.pivot(index="DATE", columns="TICKERSYMBOL", values=price_col).sort_index()

    # 월별 수익률(월초->월말)
    mret = prices.resample("ME").last().pct_change(fill_method=None)
    mret.index = mret.index.to_period("M").astype(str)  # "YYYY-MM"

    # ---- rank_panel month label normalize ----
    rp = rank_panel.copy()
    rp.index = rp.index.astype(str)

    months = rp.index.tolist()

    out = []
    for ms in months:
        # 예측 순위
        pred = rp.loc[ms].dropna()
        if pred.empty:
            out.append({"MONTH": ms, "N": 0, "IC": np.nan})
            continue

        # 비교할 실제 월(미래 lag)
        target_period = (pd.Period(ms, freq="M") + lag_months).strftime("%Y-%m")
        if target_period not in mret.index:
            out.append({"MONTH": ms, "N": 0, "IC": np.nan})
            continue

        # 실제 월수익률(미래 월)
        realized = mret.loc[target_period]

        # 공통 티커만
        common = pred.index.intersection(realized.index)
        common = common[pred.loc[common].notna() & realized.loc[common].notna()]

        if len(common) < min_n:
            out.append({"MONTH": ms, "N": int(len(common)), "IC": np.nan})
            continue

        pred_c = pred.loc[common].astype(float)

        # realized return -> realized rank (큰 수익률이 rank 1)
        realized_c = realized.loc[common].astype(float)
        realized_rank = realized_c.rank(ascending=False, method="average")

        # rank correlation
        ic = pred_c.corr(realized_rank, method=method)

        out.append({
            "MONTH": ms,
            "TARGET_MONTH": target_period,
            "N": int(len(common)),
            "IC": float(ic) if pd.notna(ic) else np.nan,
        })

    res = pd.DataFrame(out).set_index("MONTH").sort_index(key=lambda x: pd.PeriodIndex(x, freq="M"))
    return res


def rank_ic_summary(ic_df: pd.DataFrame, ic_col: str = "IC") -> pd.Series:
    x = ic_df[ic_col].dropna()

    if x.empty:
        return pd.Series({
            "mean_IC": np.nan,
            "std_IC": np.nan,
            "ICIR_annual": np.nan,
            "n_months": 0
        })

    mean_ic = x.mean()
    std_ic = x.std(ddof=0)

    icir = (mean_ic / std_ic) * np.sqrt(12) if std_ic != 0 else np.nan

    return pd.Series({
        "mean_IC": mean_ic,
        "std_IC": std_ic,
        "ICIR_annual": icir,
        "n_months": int(x.shape[0])
    })