## 📁 Cấu trúc thư mục

project/
├── app.py # Giao diện chính Streamlit
├── about_me_gen.py # Tạo 이력서 자소서 bằng KoGPT
├── about_me_evakuate.py # Đánh giá 자소서 + 이력서 bằng KoBERT
├── emotion_analyze.py # Phân tích cảm xúc từ video
├── video_pose_analyze.py # Phân tích tư thế từ OpenPose
├── 17_data_processing/ # Tạo JSON từ 이력서/자소서 (thô) -> tạo 2 files chứa toàn bộ id 1~10 đã labeling của 이력서 và 자소서 dạng json 
├── utils/ # Dùng để labeling
│ ├── 이력서_라벨링_기준.json # Tiêu chí đánh giá 이력서 (A–E)
│ ├── 자기소개서_라벨링_기준.json # Tiêu chí đánh giá 자소서 (A–E)
├── vid_frame_capture/ # Lưu frames picture và labeling vị trí cơ thể và cảm xúc từ video

## Git Tutorial

``` bash
git clone https://github.com/locdeng/resume-video-generator-andevaluator.git
```

```bash
git branch -a  # kiểm tra toàn bộ branch 
```

```bash
git checkout -b [your branch name] origin/[your branch name]  # tạo branch của mình ở local và chuyển vào làm  
```

```bash
git branch   # kiểm tra xem đang ở branch, nhay hiện tại sẽ có dấu * trước tên 
```

```bash
# Sau khi đã làm xong thì push lên lại nhánh của mình
git add .
git commit -m "your commit"
git push origin [your branch name]
```
