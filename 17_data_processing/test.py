import pandas as pd
import json

# Đọc dữ liệu JSON từ file
with open('resume-video-generator-andevaluator/17_data_processing/08_CL_data.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Hàm làm phẳng JSON lồng nhau
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '.')
        elif isinstance(x, list):
            i = 0
            for a in x:
                flatten(a, name + str(i) + '.')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

# Nếu file là danh sách các bản ghi
if isinstance(json_data, list):
    flat_data = [flatten_json(record) for record in json_data]
else:
    flat_data = [flatten_json(json_data)]

# Tạo DataFrame và lưu thành CSV
df = pd.DataFrame(flat_data)
df.to_csv('daloc.csv', index=False, encoding='utf-8-sig')
print('Đã chuyển đổi thành công sang file converted_file.csv')
