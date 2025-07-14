# config.py

# 국가공인 자격증 (A등급)
NATIONAL_CERTIFICATES = [
    "정보처리기사", "정보보안기사", "전기기사", "건축기사", 
    "토목기사", "사회복지사", "한식조리사", "위험물산업기사",
    "산업안전기사", "전산세무회계", "환경기사", "기계정비기사"
]

# 민간자격증 (C등급)
PRIVATE_CERTIFICATES = [
    "GTQ", "MOS", "컴퓨터활용능력", "코딩 자격", "마케팅 자격",
    "ERP 정보관리사", "OA Master", "그래픽 자격", "CS Leaders",
    "웹디자인 기능사", "전산응용기계제도 기능사"
]

# 유사 무자격증 (D등급)
FAKE_OR_UNCERTIFIED = [
    "수료증", "수강증", "이수증", "완료증", "참가증", 
    "기념증", "출석증", "경험증명서"
]

# 외부 전문교육 키워드
EXTERNAL_EDUCATION_KEYWORDS = [
    "전문 과정", "온라인 강의", "K-MOOC", "Udemy", "패스트캠퍼스",
    "Inflearn", "LinkedIn Learning", "Coursera", "edX", "이수", "수료"
]

# 기업 내 교육 키워드
INTERNAL_EDUCATION_KEYWORDS = [
    "사내교육", "내부교육", "워크샵", "세미나", "정기교육", 
    "신입교육", "OJT", "멘토링 프로그램"
]

# 자기개발 키워드
SELF_DEVELOPMENT_KEYWORDS = [
    "자기개발", "자기주도학습", "스터디", "독학", "자율학습", 
    "성장 노력", "개인 목표", "동기부여"
]
