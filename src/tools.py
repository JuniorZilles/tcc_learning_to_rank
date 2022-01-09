import json

def read_json(name:str):
    with open(name) as json_file:
        data = json.load(json_file)
    return data