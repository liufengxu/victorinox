import json
import requests


my_url = 'http://offline_website/compare'
my_url_get = 'http://offline_website/alive'
d = {'guid': '865555555555', 'search_id': '29111111111111'}


def simulate_post(url, para):
    data = json.dumps(para)
    response = requests.post(url=url, data=data, timeout=40, headers={'content-type': 'application/json'})
    return response.status_code, response.text


def simulate_get(url):
    response = requests.get(url=url, timeout=40)
    return response.status_code, response.text


code, text = simulate_post(my_url, d)
code, text = simulate_get(my_url_get)
print(code)
print(text)

