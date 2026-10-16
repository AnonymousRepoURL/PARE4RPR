"""Output the current time every 3 minutes, for a total of 20 lines.

python data/print_time_every_3_minutes.py
"""

import time
from datetime import datetime

INTERVAL_SECONDS = 3 * 60
MAX_LINES = 20


def main() -> None:
    for line_number in range(1, MAX_LINES + 1):
        current_time = datetime.now().astimezone()
        print(current_time.strftime("%Y-%m-%d %H:%M:%S %Z"), flush=True)

        if line_number < MAX_LINES:
            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
