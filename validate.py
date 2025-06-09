import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# — 속성 분류를 "숫자 vs. 문자열 vs. 숏핸드"로 구분 —
NUMERIC_PROPS = {
    'width', 'height', 'margin', 'padding', 'font-size',
    'border-width', 'border-radius', 'opacity', 'z-index',
    'top', 'left', 'right', 'bottom', 'max-width', 'min-width',
    'max-height', 'min-height', 'margin-left', 'margin-bottom',
    'text-indent', 'mso-font-kerning', 'mso-padding-alt',
    'page-break-before', 'line-height', 'letter-spacing',
    'margin-right', 'margin-top', 'mso-margin-bottom-alt',
    'mso-margin-top-alt'
}

STRING_PROPS = {
    'font-family', 'color', 'background-color', 'text-align', 'display',
    'position', 'float', 'clear', 'visibility', 'overflow', 'text-decoration',
    'mso-add-space', 'mso-bidi-font-family', 'mso-fareast-language',
    'mso-list', 'mso-fareast-font-family', 'background', 'border',
    'border-bottom', 'border-collapse', 'border-left', 'border-right',
    'border-top', 'mso-ansi-language', 'mso-ascii-theme-font',
    'mso-bidi-language', 'mso-bidi-theme-font', 'mso-border-alt',
    'mso-border-left-alt', 'mso-border-top-alt', 'mso-fareast-theme-font',
    'mso-hansi-theme-font', 'mso-no-proof', 'mso-special-character',
    'mso-yfti-firstrow', 'mso-yfti-irow', 'mso-yfti-lastrow',
    'mso-yfti-tbllook', 'mso-ascii-font-family', 'mso-hansi-font-family',
    'mso-pagination', 'tab-stops', 'text-autospace', 'word-break'
}

def parse_font_shorthand(value):
    out = {}
    m = re.search(r'(\d*\.?\d+)(?:px|pt|em|rem)?(?:/(\d*\.?\d+))?', value)
    if m:
        out['font-size'] = float(m.group(1))
        if m.group(2):
            out['line-height'] = float(m.group(2))
    w = re.search(r'\b([1-9]00|bold|normal)\b', value)
    if w:
        out['font-weight'] = float(w.group(1)) if w.group(1).isdigit() else w.group(1)
    s = re.search(r'\b(italic|oblique)\b', value)
    if s:
        out['font-style'] = s.group(1)
    fam = value.split()[-1]
    out['font-family'] = fam.strip().rstrip(';')
    return out

def parse_style_vector(style_str):
    style_dict = {}
    if not style_str:
        return style_dict

    for prop in [p.strip() for p in style_str.split(';') if ':' in p]:
        key, raw = [x.strip() for x in prop.split(':', 1)]
        key = key.lower()

        if key == 'font':
            style_dict.update(parse_font_shorthand(raw))
            continue

        if key in NUMERIC_PROPS:
            nums = re.findall(r'[-+]?\d*\.?\d+|\d+', raw)
            style_dict[key] = float(nums[0]) if nums else 0.0
        elif key in STRING_PROPS:
            style_dict[key] = raw

    return style_dict

def vectorize_html(html_text):
    results = []
    soup = BeautifulSoup(html_text, 'html.parser')
    for tag in soup.find_all():
        style_str = tag.get('style', '').strip()
        style_vector = parse_style_vector(style_str)
        if style_vector:
            results.append({
                'tag': str(tag).strip(),
                'style_vector': style_vector,
                'vector_length': len(style_vector)
            })
    return results

def determine_label(tag_str):
    tag_name = BeautifulSoup(tag_str, 'html.parser').find().name.lower()
    return 'heading' if re.match(r'^h[1-6]$', tag_name) else 'body'

def prepare_classifier(orig_vectors):
    dv = DictVectorizer(sparse=False)
    style_dicts = [r['style_vector'] for r in orig_vectors]
    orig_feats = dv.fit_transform(style_dicts)
    labels = [determine_label(r['tag']) for r in orig_vectors]
    return dv, orig_feats, labels

def calculate_conformity_knn(orig_feats, new_feats, k=5):
    """
    새 태그 벡터 각각에 대해, 원본 벡터들 중 유사도 상위 k개의 평균 코사인 유사도를 반환.
    """
    sim_mat = cosine_similarity(new_feats, orig_feats)
    sims = []
    for sims_to_orig in sim_mat:
        topk = np.sort(sims_to_orig)[-k:]
        sims.append(np.mean(topk))
    return np.array(sims)

def compare_html():
    orig_text = txt_orig.get('1.0', tk.END)
    new_text = txt_new.get('1.0', tk.END)

    orig_vectors = vectorize_html(orig_text)
    new_vectors = vectorize_html(new_text)

    orig_all_props = extract_all_style_properties(orig_text)
    new_all_props = extract_all_style_properties(new_text)

    orig_processed = set()
    new_processed = set()
    for r in orig_vectors:
        orig_processed.update(r['style_vector'].keys())
    for r in new_vectors:
        new_processed.update(r['style_vector'].keys())

    dv, orig_feats, labels = prepare_classifier(orig_vectors)
    new_feats = dv.transform([r['style_vector'] for r in new_vectors])

    # 태그별 유사도 계산
    conformity_scores = calculate_conformity_knn(orig_feats, new_feats, k=5)
    overall_similarity = np.mean(conformity_scores)

    log_message = "=== 스타일 속성 처리 결과 ===\n\n"
    log_message += "원본 HTML:\n"
    log_message += f"처리된 속성 ({len(orig_processed)}): {', '.join(sorted(orig_processed))}\n"
    log_message += f"처리되지 않은 속성 ({len(orig_all_props - orig_processed)}): {', '.join(sorted(orig_all_props - orig_processed))}\n\n"
    log_message += "생성된 HTML:\n"
    log_message += f"처리된 속성 ({len(new_processed)}): {', '.join(sorted(new_processed))}\n"
    log_message += f"처리되지 않은 속성 ({len(new_all_props - new_processed)}): {', '.join(sorted(new_all_props - new_processed))}\n\n"
    log_message += f"전체 유사도 (태그별 평균): {overall_similarity:.4f}\n"

    txt_log.config(state='normal')
    txt_log.delete('1.0', tk.END)
    txt_log.insert(tk.END, log_message)
    txt_log.config(state='disabled')

    # (기존 compare_html 에서 결과 텍스트 채우기와 plot_vectors 호출 부분은 그대로 유지)
    orig_lines = []
    for r in orig_vectors:
        vector_str = ', '.join(
            f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: '{v}'"
            for k, v in r['style_vector'].items()
        )
        orig_lines.append(
            f"{r['tag']}\n"
            f"  스타일 벡터: [{vector_str}]\n"
            f"  벡터 차원: {r['vector_length']}"
        )

    new_lines = []
    new_labels = []
    for feat, r in zip(new_feats, new_vectors):
        dists = np.linalg.norm(orig_feats - feat, axis=1)
        role = labels[int(np.argmin(dists))]
        new_labels.append(role)
        vector_str = ', '.join(
            f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: '{v}'"
            for k, v in r['style_vector'].items()
        )
        new_lines.append(
            f"{r['tag']}\n"
            f"  스타일 벡터: [{vector_str}]\n"
            f"  벡터 차원: {r['vector_length']}\n"
            f"  분류 결과: {role}"
        )

    txt_orig_result.config(state='normal')
    txt_orig_result.delete('1.0', tk.END)
    txt_orig_result.insert(tk.END, "\n\n".join(orig_lines))
    txt_orig_result.config(state='disabled')

    txt_new_result.config(state='normal')
    txt_new_result.delete('1.0', tk.END)
    txt_new_result.insert(tk.END, "\n\n".join(new_lines))
    txt_new_result.config(state='disabled')

    plot_vectors(orig_feats, labels, new_feats, new_labels, overall_similarity)

def extract_all_style_properties(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    all_props = set()
    for tag in soup.find_all():
        style_str = tag.get('style', '').strip()
        if style_str:
            props = [p.strip().split(':')[0].strip().lower()
                     for p in style_str.split(';') if ':' in p]
            all_props.update(props)
    return all_props

def show_property_similarity():
    orig_text = txt_orig.get('1.0', tk.END)
    new_text = txt_new.get('1.0', tk.END)

    orig_vectors = vectorize_html(orig_text)
    new_vectors = vectorize_html(new_text)
    
    similarities, details = calculate_property_similarity(orig_vectors, new_vectors)
    
    # 결과 창 생성
    result_window = tk.Toplevel(root)
    result_window.title("속성별 유사도")
    result_window.geometry("800x600")
    
    left_frame = ttk.Frame(result_window)
    right_frame = ttk.Frame(result_window)
    left_frame.pack(side="left", fill="both", expand=True)
    right_frame.pack(side="right", fill="both", expand=True)
    
    tree = ttk.Treeview(left_frame, columns=("속성", "유사도"), show="headings")
    tree.heading("속성", text="속성")
    tree.heading("유사도", text="유사도")
    scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    for prop, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        tree.insert("", "end", values=(prop, f"{sim:.4f}"))
    
    detail_text = scrolledtext.ScrolledText(right_frame, width=50, height=30)
    detail_text.pack(fill="both", expand=True)
    
    def on_select(event):
        sel = tree.selection()
        if not sel: return
        prop = tree.item(sel[0])['values'][0]
        info = details[prop]
        detail_text.delete('1.0', tk.END)
        for k, v in info.items():
            detail_text.insert(tk.END, f"{k}: {v}\n")
    
    tree.bind('<<TreeviewSelect>>', on_select)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

def calculate_property_similarity(orig_vectors, new_vectors):
    all_props = set()
    for vec in orig_vectors + new_vectors:
        all_props.update(vec['style_vector'].keys())
    similarities = {}
    details = {}
    for prop in all_props:
        orig_vals = [v['style_vector'][prop] for v in orig_vectors if prop in v['style_vector']]
        new_vals = [v['style_vector'][prop] for v in new_vectors if prop in v['style_vector']]
        if prop in NUMERIC_PROPS:
            if orig_vals and new_vals:
                om = np.mean(orig_vals)
                nm = np.mean(new_vals)
                sim = 1 - abs(om - nm) / (max(om, nm) + 1e-8)
                similarities[prop] = sim
                details[prop] = {
                    'type':'numeric',
                    'orig_mean':om, 'new_mean':nm,
                    'calculation':f"1 - |{om:.2f}-{nm:.2f}|/max({om:.2f},{nm:.2f})"
                }
        else:
            if orig_vals and new_vals:
                os, ns = set(orig_vals), set(new_vals)
                inter = len(os & ns)
                uni = len(os | ns)
                sim = inter/uni if uni else 0
                similarities[prop] = sim
                details[prop] = {
                    'type':'string',
                    'intersection':list(os&ns),
                    'union':list(os|ns),
                    'calculation':f"{inter}/{uni}"
                }
    return similarities, details

def show_tag_similarity():
    orig_text = txt_orig.get('1.0', tk.END)
    new_text = txt_new.get('1.0', tk.END)

    orig_vectors = vectorize_html(orig_text)
    new_vectors = vectorize_html(new_text)

    # DictVectorizer & features 생성
    dv, orig_feats, labels = prepare_classifier(orig_vectors)
    new_feats = dv.transform([r['style_vector'] for r in new_vectors])

    # k-NN 부합도 계산 (k=5)
    conformity_scores = calculate_conformity_knn(orig_feats, new_feats, k=5)

    # 결과 창
    result_window = tk.Toplevel(root)
    result_window.title("태그별 부합도 (k-NN)")
    result_window.geometry("900x600")

    left_frame = ttk.Frame(result_window)
    right_frame = ttk.Frame(result_window)
    left_frame.pack(side="left", fill="both", expand=True)
    right_frame.pack(side="right", fill="both", expand=True)

    # 왼쪽: 태그 & 부합도 목록
    tree = ttk.Treeview(left_frame, columns=("태그","부합도"), show="headings")
    tree.heading("태그", text="태그 (HTML)")
    tree.heading("부합도", text="부합도")
    scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    details_map = {}
    for idx, vec in enumerate(new_vectors):
        tag_html = vec['tag']
        score = conformity_scores[idx]
        tree.insert("", "end", values=(tag_html, f"{score:.4f}"))
        details_map[tag_html] = (vec['style_vector'], score)

    detail_text = scrolledtext.ScrolledText(right_frame, width=50, height=30)
    detail_text.pack(fill="both", expand=True)

    def on_select(event):
        sel = tree.selection()
        if not sel: return
        tag_html, score_str = tree.item(sel[0])['values']
        style_vec, score = details_map[tag_html]
        detail_text.delete('1.0', tk.END)
        detail_text.insert(tk.END, f"태그:\n{tag_html}\n\n")
        detail_text.insert(tk.END, f"부합도: {score:.4f}\n\n")
        detail_text.insert(tk.END, "스타일 벡터:\n")
        for k, v in style_vec.items():
            val = f"{v:.2f}" if isinstance(v, float) else f"'{v}'"
            detail_text.insert(tk.END, f"  {k}: {val}\n")

    tree.bind('<<TreeviewSelect>>', on_select)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

def load_file():
    try:
        with open('existed_html.txt', 'r', encoding='utf-8') as f:
            txt_orig.delete('1.0', tk.END)
            txt_orig.insert(tk.END, f.read())
    except Exception as e:
        messagebox.showerror("오류", f"원본 HTML 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
    try:
        with open('valid_generated_html.txt', 'r', encoding='utf-8') as f:
            txt_new.delete('1.0', tk.END)
            txt_new.insert(tk.END, f.read())
    except Exception as e:
        messagebox.showerror("오류", f"생성된 HTML 파일을 읽는 중 오류가 발생했습니다: {str(e)}")

def plot_vectors(orig_feats, labels, new_feats, new_labels, overall_similarity):
    # PCA로 2차원으로 차원 축소
    pca = PCA(n_components=2)
    all_feats = np.vstack([orig_feats, new_feats])
    reduced = pca.fit_transform(all_feats)
    
    # 원본과 새로운 데이터 분리
    n_orig = len(orig_feats)
    orig_reduced = reduced[:n_orig]
    new_reduced = reduced[n_orig:]
    
    # 산점도 그리기
    plt.figure(figsize=(10, 8))
    
    # 원본 데이터 (heading/body 구분)
    for label in ['heading', 'body']:
        mask = [l == label for l in labels]
        plt.scatter(orig_reduced[mask, 0], orig_reduced[mask, 1], 
                   label=f'Original {label}', alpha=0.6)
    
    # 새로운 데이터 (heading/body 구분)
    for label in ['heading', 'body']:
        mask = [l == label for l in new_labels]
        plt.scatter(new_reduced[mask, 0], new_reduced[mask, 1], 
                   label=f'New {label}', marker='^', alpha=0.6)
    
    plt.title(f'Style Vector PCA Visualization (Overall Similarity: {overall_similarity:.4f})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

root = tk.Tk()
root.title("HTML 인라인 스타일 벡터화 및 k-NN 태그 부합도 시각화")

tk.Label(root, text="원본 HTML").grid(row=0, column=0, padx=5, pady=5)
txt_orig = scrolledtext.ScrolledText(root, width=50, height=15)
txt_orig.grid(row=1, column=0, padx=5, pady=5)

tk.Label(root, text="생성 HTML").grid(row=0, column=1, padx=5, pady=5)
txt_new = scrolledtext.ScrolledText(root, width=50, height=15)
txt_new.grid(row=1, column=1, padx=5, pady=5)

btn_load = tk.Button(root, text="파일 로드", command=load_file)
btn_load.grid(row=2, column=0, columnspan=2, pady=5)

btn = tk.Button(root, text="벡터화·분류·시각화 시작", command=compare_html)
btn.grid(row=3, column=0, columnspan=2, pady=5)

btn_similarity = tk.Button(root, text="속성별 유사도 확인", command=show_property_similarity)
btn_similarity.grid(row=4, column=0, columnspan=2, pady=5)

btn_tag_similarity = tk.Button(root, text="태그별 부합도 확인", command=show_tag_similarity)
btn_tag_similarity.grid(row=5, column=0, columnspan=2, pady=5)

tk.Label(root, text="스타일 속성 처리 로그").grid(row=6, column=0, columnspan=2, padx=5, pady=5)
txt_log = scrolledtext.ScrolledText(root, width=100, height=10, state='disabled')
txt_log.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

tk.Label(root, text="원본 벡터화 결과").grid(row=8, column=0, padx=5, pady=5)
txt_orig_result = scrolledtext.ScrolledText(root, width=50, height=20, state='disabled')
txt_orig_result.grid(row=9, column=0, padx=5, pady=5)

tk.Label(root, text="변경된 벡터화 결과").grid(row=8, column=1, padx=5, pady=5)
txt_new_result = scrolledtext.ScrolledText(root, width=50, height=20, state='disabled')
txt_new_result.grid(row=9, column=1, padx=5, pady=5)

root.after(100, load_file)
root.mainloop()
