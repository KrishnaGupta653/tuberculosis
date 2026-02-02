import requests

# The path to the image you want to test
image_path = "TB.1.jpg" 

url = "http://localhost:5000/predict"

with open(image_path, 'rb') as f:
    response = requests.post(url, files={'file': f})

print(response.json())