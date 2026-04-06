import requests
import time
import numpy as np
import json
import concurrent.futures

FASTAPI_URL = "http://localhost:8000/predict"

with open("input_example.json", "r") as f:
    payload = json.load(f)

def send_request(payload):
    start = time.time()
    resp = requests.post(FASTAPI_URL, json=payload)
    return time.time() - start if resp.status_code == 200 else None

def print_metrics(times, num_reqs, total_time=None):
    times = np.array(times)
    throughput = num_reqs / (total_time if total_time else times.sum())
    print(f"Median: {np.median(times):.4f}s | 95th: {np.percentile(times, 95):.4f}s | Throughput: {throughput:.2f} req/s")

def run_test(num_reqs, workers=1):
    print(f"\n--- Test: {num_reqs} requests, {workers} workers ---")
    times = []
    failures = 0  # Add a failure counter
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_reqs)]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            if result is not None: 
                times.append(result)
            else:
                failures += 1 # Count failed HTTP requests
                
    total_time = time.time() - start_total
    print_metrics(times, num_reqs, total_time)
    
    # Calculate and print the error rate
    error_rate = (failures / num_reqs) * 100
    print(f"System Error Rate: {error_rate:.2f}% ({failures} failed requests)")

if __name__ == "__main__":
    print("Warming up...")
    requests.post(FASTAPI_URL, json=payload)
    run_test(num_reqs=20, workers=1)  # Sequential test
    run_test(num_reqs=50, workers=15)  # Concurrent test (Peak load)
