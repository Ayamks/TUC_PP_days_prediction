import argparse
from src.predict import predict


def parse_args():
    parser = argparse.ArgumentParser(description="PP1/PP2 daily prediction from weather data")
    parser.add_argument("--input", required=True, help="Path to input hourly weather CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV with predictions")
    parser.add_argument("--start", default="2026-02-20", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-26", help="End date (YYYY-MM-DD)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        input_weather_csv=args.input,
        output_csv=args.output,
        start_date=args.start,
        end_date=args.end,
    )
