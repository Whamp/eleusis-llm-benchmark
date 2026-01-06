import os
import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


## COMPLETION

model = "openai/gpt-4o-mini"

print(f"TESTING COMPLETION ({model})")

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
    "max_tokens": 10,
}

resp = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"Response: {content}")
    print(f"Tokens - prompt: {usage.get('prompt_tokens')}, completion: {usage.get('completion_tokens')}")
else:
    print(f"Error: {resp.json()}")


## CREDIT CHECK

print("CHECKING CREDITS")

resp = requests.get(f"{BASE_URL}/credits", headers=HEADERS)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")

if resp.status_code == 200:
    data = resp.json().get("data", {})
    total = data.get("total_credits", 0)
    used = data.get("total_usage", 0)
    remaining = total - used
    print(f"\nTotal credits:  ${total:.4f}")
    print(f"Used:           ${used:.4f}")
    print(f"Remaining:      ${remaining:.4f}")