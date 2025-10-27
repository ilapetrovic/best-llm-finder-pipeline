import subprocess
import time
import csv
from datetime import datetime

LOG_FILE = "gpu_usage_log.csv"
INTERVAL = 5  # seconds

def get_gpu_stats():
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip().splitlines()

def main():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "gpu_name", "gpu_util%", "mem_util%",
            "mem_used_MB", "mem_total_MB", "temp_C"
        ])
        print(f"[{datetime.now()}] GPU monitor started. Logging every {INTERVAL}s...")
        try:
            while True:
                stats = get_gpu_stats()
                for line in stats:
                    writer.writerow(line.split(", "))
                f.flush()
                time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
