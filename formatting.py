import json

# # Load the JSON file
# with open('img_emotion_training_data(기쁨).json', 'r', encoding='UTF-8') as file:
#     data = json.load(file)

# # Prettify and write back to a new file
# with open('img_emotion_training_data(기쁨).json', 'w') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4, sort_keys=True)
    
# # converting euc-kr to utf-8 encoding
# # with open('img_emotion_training_data(기쁨).json', 'r', encoding='euc-kr') as file:
# #     euc_kr = file.read()
# #     with open('img_emotion_training_data(기쁨)-utf8.json', 'w', encoding='utf-8') as file_utf8:
# #         file_utf8.write(euc_kr)
    

# print("JSON file has been prettified and saved as 'pretty_img_emotion_training_data(기쁨).json'")

with open('img_emotion_training_data(분노).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('img_emotion_training_data(분노).json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)