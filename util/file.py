import json

def load_json(path):
    """
    读取 JSON 文件并返回 Python 对象（dict / list）
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def dump_json(data, path):
    """
    将 Python 对象（dict / list）写入 JSON 文件
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_jsonl(path):
    """
    读取 JSONL 文件，每行一个 JSON。
    返回 Python 列表（list[dict]）。
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def dump_jsonl(data, path):
    """
    将 list[dict] 输出为 JSONL 文件，每行一个 JSON。
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")