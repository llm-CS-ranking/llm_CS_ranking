from __future__ import annotations

import pandas as pd

from llm_ranking.paths import NDX_DATA_DIR


def load_market() -> tuple[pd.DataFrame, dict[int, str], pd.DataFrame]:
    trading = pd.read_csv(NDX_DATA_DIR / "ndx_tradingiteminfo.csv")
    member = pd.read_csv(NDX_DATA_DIR / "ndx_data_member.csv", parse_dates=["DATE"])
    market = pd.read_csv(NDX_DATA_DIR / "ndx_market_data.csv", parse_dates=["DATE"])

    latest_tid = (
        trading.sort_values(["TICKERSYMBOL", "TRADINGITEMID"])
        .drop_duplicates("TICKERSYMBOL", keep="last")
    )
    tid2ticker = dict(zip(latest_tid["TRADINGITEMID"], latest_tid["TICKERSYMBOL"]))
    cid2ticker = dict(zip(latest_tid["COMPANYID"], latest_tid["TICKERSYMBOL"]))

    market = market.loc[market["TRADINGITEMID"].isin(tid2ticker)].copy()
    market["TICKERSYMBOL"] = market["TRADINGITEMID"].map(tid2ticker)
    market = market.drop_duplicates(subset=["DATE", "TICKERSYMBOL"], keep="last")
    market = market.set_index(["DATE", "TICKERSYMBOL"]).sort_index()

    member = member.loc[member["TRADINGITEMID"].isin(tid2ticker)].copy()
    member["TICKERSYMBOL"] = member["TRADINGITEMID"].map(tid2ticker)
    return market, cid2ticker, member


def load_fundamentals() -> pd.DataFrame:
    return pd.read_csv(NDX_DATA_DIR / "ndx_fundamental_data.csv")

