import pandas as pd
import os
import time
from datetime import datetime

# Chromecast device name
CHROMECAST_NAME = "Media"  # Update with your Google Home device name
CSV_FILE_PATH = "/srv/storage/custom-tools/Adhan/prayer_times_2026_final.csv"

def get_next_audio():
    """Finds the next audio file and the exact minute to play."""
    df = pd.read_csv(CSV_FILE_PATH)

    # Ensure correct column names (case insensitive)
    df.columns = df.columns.str.lower()

    # Verify required columns exist
    required_columns = {"date", "time", "file_location"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    # Convert date and time to datetime
    df["date_time"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")

    # Get current timestamp
    now = datetime.now()
    
    # Find the next scheduled time (greater than or equal to current time)
    upcoming = df[df["date_time"] >= now].sort_values("date_time").head(1)

    if not upcoming.empty:
        scheduled_time = upcoming.iloc[0]["date_time"]
        file_location = upcoming.iloc[0]["file_location"]
        return scheduled_time, file_location

    return None, None  # No upcoming schedule

def play_audio(file_location):
    """Plays the audio file using catt."""
    print(f"Playing: {file_location}")
    os.system(f'/srv/storage/custom-tools/python-venv/bin/catt -d "{CHROMECAST_NAME}" cast "{file_location}"')

# Main Execution
scheduled_time, audio_file = get_next_audio()

if scheduled_time and audio_file:
    now = datetime.now()
    wait_time = (scheduled_time - now).total_seconds()
    
    print(f"Next prayer call is after {wait_time:.2f} seconds")
    
    if wait_time > 3600:
        print("Skipping execution because next prayer call is more than 1 hour.")
        exit(0)  # Exit the script

    if wait_time > 0 and wait_time <= 3600:
        print(f"Waiting for {wait_time:.2f} seconds until {scheduled_time.strftime('%H:%M')}")
        time.sleep(wait_time)  # Sleep until the exact scheduled time

    play_audio(audio_file)
