import requests

print("Sending request...")
response = requests.post(
    "http://127.0.0.1:5000/upload",
    files={"file": ("test.txt", "This is a test of the USA. Christmas tree rash.")},
    data={"direction": "en-tr"},
    stream=True
)

print(f"Status Code: {response.status_code}")
print("Headers:", response.headers)

for line in response.iter_lines():
    if line:
        print(f"Received: {line.decode('utf-8')[:100]}")
