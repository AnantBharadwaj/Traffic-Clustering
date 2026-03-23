import requests
import os

API_KEY = os.getenv("TOMTOM_API_KEY", "")


def fetch_test_flow(point="17.3850,78.4867"):
	if not API_KEY:
		raise ValueError("TOMTOM_API_KEY is not set in environment variables")

	url = (
		"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
		f"?point={point}&key={API_KEY}"
	)
	response = requests.get(url, timeout=15)
	response.raise_for_status()
	return response.json()


if __name__ == "__main__":
	print(fetch_test_flow())