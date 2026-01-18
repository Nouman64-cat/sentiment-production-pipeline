import os
import requests

DATA_URL = "https://ai-ml-dl---datasets.s3.us-east-1.amazonaws.com/test-task-dataset/data.csv"
OUTPUT_DIR = os.path.join("src", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data.csv")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Downloading from {DATA_URL}")

    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()

        with open(OUTPUT_FILE, "wb") as f:
            f.write(response.content)

        print(f"Saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Failed to download: {e}")
        raise

if __name__ == "__main__":
    main()