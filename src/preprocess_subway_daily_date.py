"""subway_daily 분석용 데이터셋의 날짜 기준을 생성합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample" / "raw_subway_daily_sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sample" / "processed_subway_daily_sample.csv"

REQUIRED_COLUMNS = [
    "사용일자",
    "노선명",
    "역명",
    "승차총승객수",
    "하차총승객수",
]

OUTPUT_COLUMNS = [
    "date",
    "year",
    "month",
    "day",
    "date_type",
    "line_name",
    "station_name",
    "daily_in_passengers",
    "daily_out_passengers",
    "daily_total_passengers",
]

LINE_NAME_MAPPING = {
    "경의선": "경의중앙선",
    "공항철도 1호선": "공항철도",
}


def preprocess_subway_daily(data: pd.DataFrame) -> pd.DataFrame:
    """Raw Data를 subway_daily 분석용 컬럼 구조로 변환합니다."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "입력 데이터에 필요한 컬럼이 없습니다: " + ", ".join(missing_columns)
        )

    raw_dates = data["사용일자"].astype("string").str.strip()
    invalid_format = raw_dates.isna() | ~raw_dates.str.fullmatch(r"\d{8}", na=False)

    if invalid_format.any():
        invalid_rows = (data.index[invalid_format] + 2).tolist()
        raise ValueError(
            "'사용일자'는 YYYYMMDD 형식의 8자리 값이어야 합니다. "
            f"확인이 필요한 CSV 행: {invalid_rows}"
        )

    try:
        parsed_dates = pd.to_datetime(raw_dates, format="%Y%m%d", errors="raise")
    except ValueError as error:
        raise ValueError(
            "'사용일자'에 실제 달력에 존재하지 않는 날짜가 포함되어 있습니다."
        ) from error

    passenger_columns = ["승차총승객수", "하차총승객수"]
    passengers = data[passenger_columns].apply(pd.to_numeric, errors="coerce")
    invalid_passengers = passengers.isna().any(axis=1)

    if invalid_passengers.any():
        invalid_rows = (data.index[invalid_passengers] + 2).tolist()
        raise ValueError(
            "승하차 인원은 숫자여야 합니다. "
            f"확인이 필요한 CSV 행: {invalid_rows}"
        )

    station_names = data["역명"].astype("string").str.strip()
    station_names = station_names.where(
        station_names.str.endswith("역", na=False),
        station_names + "역",
    )
    line_names = data["노선명"].astype("string").str.strip()
    line_names = line_names.replace(LINE_NAME_MAPPING)

    result = pd.DataFrame(
        {
            "date": parsed_dates.dt.strftime("%Y-%m-%d"),
            "year": parsed_dates.dt.year,
            "month": parsed_dates.dt.month,
            "day": parsed_dates.dt.day,
            "date_type": parsed_dates.dt.dayofweek.ge(5).map(
                {True: "주말", False: "평일"}
            ),
            "line_name": line_names,
            "station_name": station_names,
            "daily_in_passengers": passengers["승차총승객수"].astype("int64"),
            "daily_out_passengers": passengers["하차총승객수"].astype("int64"),
        }
    )
    result["daily_total_passengers"] = (
        result["daily_in_passengers"] + result["daily_out_passengers"]
    )

    return (
        result[OUTPUT_COLUMNS]
        .sort_values(["date", "line_name", "station_name"], kind="stable")
        .reset_index(drop=True)
    )


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """공공데이터에서 자주 쓰이는 UTF-8과 CP949 인코딩을 순서대로 읽습니다."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"사용일자": "string"})
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", dtype={"사용일자": "string"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="subway_daily 분석용 데이터셋의 날짜 기준을 생성합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"입력 CSV 경로 (기본값: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"출력 CSV 경로 (기본값: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    raw_data = read_csv_with_fallback(input_path)
    processed_data = preprocess_subway_daily(raw_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"전처리 완료: {len(processed_data):,}행")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()
