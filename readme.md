
# HTML 인라인 스타일 벡터화 및 k-NN 태그 부합도 툴

## 1. 개요
- `tkinter` 기반 GUI 애플리케이션으로 인라인 스타일이 적용된 HTML 스니펫 비교
- `BeautifulSoup`을 사용해 HTML 파싱 및 인라인 스타일 속성 추출
- CSS 선언을 숫자 및 문자열 특징 벡터로 변환
- 머신러닝 기법(`DictVectorizer`, k-NN, PCA)을 통해 스타일 일치도 평가 및 시각화

## 2. 스타일 파싱
### 숫자형 및 문자열형 속성 분류
- **NUMERIC_PROPS**: 숫자 값으로 파싱하는 CSS 속성 (예: `width`, `font-size`, `opacity`)
- **STRING_PROPS**: 범주형 문자열로 처리하는 CSS 속성 (예: `color`, `display`, `font-family`)

### `parse_font_shorthand(value)`
- CSS `font` 축약 속성에서 `font-size`, 선택적 `line-height`, `font-weight`, `font-style`, `font-family` 추출

### `parse_style_vector(style_str)`
- 인라인 스타일 문자열을 `속성: 값` 쌍으로 분리
- 숫자형 속성: 수치 값 추출 및 실수(float)로 저장
- 문자열형 속성: 원시 문자열 값 저장
- `font` 축약 속성은 `parse_font_shorthand`로 위임하여 추가 속성 파싱

## 3. HTML 벡터화
- **`vectorize_html(html_text)`**:
  - BeautifulSoup으로 HTML 파싱
  - 각 태그에서 스타일 벡터 생성:
    - 원본 태그 HTML (`tag`)
    - 파싱된 스타일 특성 벡터 (`style_vector`)
    - 특성 개수 (`vector_length`)
- **`extract_all_style_properties(html_text)`**:
  - 모든 태그에서 등장한 CSS 속성 이름을 한 곳에 수집

## 4. 분류
- **`determine_label(tag_str)`**:
  - `<h1>–<h6>` 태그를 `heading`, 그 외는 `body`로 레이블링
- **`prepare_classifier(orig_vectors)`**:
  - `DictVectorizer`로 스타일 벡터를 머신러닝용 특징 행렬로 변환
  - 원본 벡터의 레이블 리스트 생성

## 5. 유사도 계산
### 태그별 부합도 (k-NN)
- **`calculate_conformity_knn(orig_feats, new_feats, k=5)`**:
  - 새 벡터와 원본 벡터 간 코사인 유사도 계산
  - 각 새 벡터에 대해 유사도 상위 k개 평균값 사용

### 속성별 유사도
- **`calculate_property_similarity(orig_vectors, new_vectors)`**:
  - **숫자형 속성**: 유사도 = `1 − |mean₁ − mean₂| / max(mean₁, mean₂)`
  - **문자열형 속성**: 유사도 = `|교집합| / |합집합|`
  - 속성별 유사도 값과 계산 상세 정보를 반환

## 6. 시각화
### PCA 산점도
- **`plot_vectors(orig_feats, labels, new_feats, new_labels, overall_similarity)`**:
  - PCA로 2차원으로 차원 축소
  - 원본(`o`)과 새 데이터(`^`)를 `heading`/`body`별로 플롯
  - 전체 유사도를 플롯 제목에 표시

### GUI 인터랙션
- **메인 윈도우 구성**:
  - 원본 및 생성 HTML용 `ScrolledText` 위젯
  - 버튼:
    - **파일 로드**: `existed_html.txt` 및 `valid_generated_html.txt` 불러오기
    - **벡터화·분류·시각화 시작**: 파싱 → 분류 → 유사도 계산 → 시각화 순으로 실행
    - **속성별 유사도 확인**: `Treeview`로 속성별 유사도 표시 창 열기
    - **태그별 부합도 확인**: `Treeview`로 태그별 부합도 표시 창 열기
- **결과 표시**:
  - 처리된/처리되지 않은 속성 로그
  - 원본 및 새 벡터 결과를 텍스트 영역에 상세히 출력

## 7. 의존성
- `tkinter`, `scrolledtext`, `ttk`
- `BeautifulSoup`(`bs4`)
- `re`, `numpy`, `matplotlib`, `sklearn`(`PCA`, `DictVectorizer`, `cosine_similarity`)
