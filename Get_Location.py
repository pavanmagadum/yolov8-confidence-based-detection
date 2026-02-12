import requests
import json

def get_location_by_ip():
    try:
        response = requests.get('https://ipinfo.io/json')
        
        if response.status_code == 200:
            data = response.json()

            print(f"IP Address: {data.get('ip')}")
            print(f"City: {data.get('city')}")
            print(f"Region: {data.get('region')}")
            print(f"Country: {data.get('country_name')}")
            print(f"Latitude: {data.get('lat')}")
            print(f"Longitude: {data.get('lon')}")
            print(f"data: {data}")
            return data
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None
if __name__ == "__main__":
    get_location_by_ip()
