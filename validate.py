import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import os

# — 속성 분류를 "숫자 vs. 문자열 vs. 축약형"으로 구분 —
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
    'background', 'border', 'border-bottom', 'border-collapse', 'border-left',
    'border-right', 'border-top', 'tab-stops', 'text-autospace', 'word-break'
}

# MSO 속성 맵핑 테이블
MSO_TO_NORMAL = {
    'mso-margin-top-alt': 'margin-top',
    'mso-margin-bottom-alt': 'margin-bottom',
    'mso-ascii-font-family': 'font-family',
    'mso-hansi-font-family': 'font-family',
    'mso-bidi-font-family': 'font-family',
    'mso-border-alt': 'border',
    'mso-border-left-alt': 'border-left',
    'mso-border-top-alt': 'border-top',
    'mso-padding-alt': 'padding',
    'mso-font-kerning': 'font-kerning'
}

# font 축약형 파싱
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

# padding/margin 축약형 파싱 (top/right/bottom/left)
def parse_box_shorthand(raw, prop):
    nums = re.findall(r'[-+]?\d*\.?\d+', raw)
    nums = [float(n) for n in nums]
    if len(nums) == 1:
        vals = nums * 4
    elif len(nums) == 2:
        vals = [nums[0], nums[1], nums[0], nums[1]]
    elif len(nums) == 3:
        vals = [nums[0], nums[1], nums[2], nums[1]]
    else:
        vals = nums[:4]
    sides = ['top', 'right', 'bottom', 'left']
    return {f"{prop}-{side}": val for side, val in zip(sides, vals)}

# 인라인 스타일 문자열 → 속성 벡터 딕셔너리
def parse_style_vector(style_str):
    style_dict = {}
    if not style_str:
        return style_dict
    for prop in [p.strip() for p in style_str.split(';') if ':' in p]:
        key, raw = [x.strip() for x in prop.split(':', 1)]
        key = key.lower()
        # MSO 속성 매핑
        if key in MSO_TO_NORMAL:
            key = MSO_TO_NORMAL[key]
        elif key.startswith('mso-'):
            key = key.replace('mso-', '')
        # font shorthand
        if key == 'font':
            style_dict.update(parse_font_shorthand(raw))
            continue
        # padding/margin shorthand
        if key in ('padding', 'margin'):
            style_dict.update(parse_box_shorthand(raw, key))
            continue
        # numeric
        if key in NUMERIC_PROPS:
            nums = re.findall(r'[-+]?\d*\.?\d+', raw)
            style_dict[key] = float(nums[0]) if nums else 0.0
        # string
        elif key in STRING_PROPS:
            style_dict[key] = raw
    return style_dict

# HTML 전체 → 태그별 스타일 벡터 리스트
def vectorize_html(html_text):
    results = []
    soup = BeautifulSoup(html_text, 'html.parser')
    for tag in soup.find_all():
        style_str = tag.get('style', '').strip()
        vec = parse_style_vector(style_str)
        if vec:
            results.append({'tag': str(tag).strip(), 'style_vector': vec, 'vector_length': len(vec)})
    return results

# 태그 이름에 따라 레이블 지정
def determine_label(tag_str):
    name = BeautifulSoup(tag_str, 'html.parser').find().name.lower()
    return 'heading' if re.match(r'^h[1-6]$', name) else 'body'

# DictVectorizer로 피처 생성 및 레이블
def prepare_classifier(vectors):
    dv = DictVectorizer(sparse=False)
    feats = dv.fit_transform([v['style_vector'] for v in vectors])
    labels = [determine_label(v['tag']) for v in vectors]
    return dv, feats, labels

# k-NN 기반 부합도 계산
def calculate_conformity_knn(orig_feats, new_feats, k=5):
    sim_mat = cosine_similarity(new_feats, orig_feats)
    return np.array([np.mean(np.sort(row)[-k:]) for row in sim_mat])

# HTML 비교 및 GUI 업데이트
def compare_html():
    orig = txt_orig.get('1.0', tk.END)
    new = txt_new.get('1.0', tk.END)
    orig_vec = vectorize_html(orig)
    new_vec = vectorize_html(new)
    dv, orig_feats, labels = prepare_classifier(orig_vec)
    new_feats = dv.transform([v['style_vector'] for v in new_vec])
    # 수치형 피처 스케일링
    scaler = StandardScaler()
    names = dv.get_feature_names_out()
    num_idx = [i for i,f in enumerate(names) if f in NUMERIC_PROPS]
    if num_idx:
        orig_feats[:, num_idx] = scaler.fit_transform(orig_feats[:, num_idx])
        new_feats[:, num_idx] = scaler.transform(new_feats[:, num_idx])
    # 유사도 계산
    scores = calculate_conformity_knn(orig_feats, new_feats, k=5)
    overall = np.mean(scores)
    # 로그 작성
    orig_props = extract_all_style_properties(orig)
    new_props = extract_all_style_properties(new)
    proc_orig = set().union(*(v['style_vector'].keys() for v in orig_vec))
    proc_new = set().union(*(v['style_vector'].keys() for v in new_vec))
    msg = "=== 스타일 속성 처리 결과 ===\n\n"
    msg += f"원본 처리된: {len(proc_orig)} → {', '.join(sorted(proc_orig))}\n"
    msg += f"원본 미처리: {len(orig_props-proc_orig)} → {', '.join(sorted(orig_props-proc_orig))}\n\n"
    msg += f"생성 처리된: {len(proc_new)} → {', '.join(sorted(proc_new))}\n"
    msg += f"생성 미처리: {len(new_props-proc_new)} → {', '.join(sorted(new_props-proc_new))}\n\n"
    msg += f"전체 유사도: {overall:.4f}\n"
    txt_log.config(state='normal'); txt_log.delete('1.0', tk.END); txt_log.insert(tk.END, msg); txt_log.config(state='disabled')
    # 결과 텍스트 패널 업데이트
    def format_vec(v): return ', '.join(f"{k}: {v:.2f}" if isinstance(v,float) else f"{k}: '{v}'" for k,v in v.items())
    txt_orig_result.config(state='normal'); txt_orig_result.delete('1.0', tk.END)
    txt_orig_result.insert(tk.END, '\n\n'.join(f"{r['tag']}\n  벡터: [{format_vec(r['style_vector'])}]" for r in orig_vec)); txt_orig_result.config(state='disabled')
    txt_new_result.config(state='normal'); txt_new_result.delete('1.0', tk.END)
    new_labels = []
    for feat, r in zip(new_feats, new_vec):
        dists = np.linalg.norm(orig_feats-feat,axis=1)
        role = labels[int(np.argmin(dists))]
        new_labels.append(role)
    txt_new_result.insert(tk.END, '\n\n'.join(f"{r['tag']}\n  벡터: [{format_vec(r['style_vector'])}]\n  분류: {label}" for r,label in zip(new_vec,new_labels)))
    txt_new_result.config(state='disabled')
    # 시각화
    plot_vectors(orig_feats, labels, new_feats, new_labels, overall)

# 모든 style 속성 추출
def extract_all_style_properties(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    props = set()
    for tag in soup.find_all():
        for p in tag.get('style','').split(';'):
            if ':' in p: props.add(p.split(':')[0].strip().lower())
    return props

# 속성별 유사도 계산 & 상세보기
def calculate_property_similarity(orig_vec, new_vec):
    all_props = set().union(*(v['style_vector'].keys() for v in orig_vec+new_vec))
    sims, details = {}, {}
    for prop in all_props:
        orig_vals = [v['style_vector'][prop] for v in orig_vec if prop in v['style_vector']]
        new_vals = [v['style_vector'][prop] for v in new_vec if prop in v['style_vector']]
        if prop in NUMERIC_PROPS:
            if orig_vals and new_vals:
                om, nm = np.mean(orig_vals), np.mean(new_vals)
                sim = 1-abs(om-nm)/(max(om,nm)+1e-8)
                sims[prop], details[prop] = sim, {'type':'numeric','orig_mean':om,'new_mean':nm}
        else:
            if orig_vals and new_vals:
                os, ns = set(orig_vals), set(new_vals)
                inter, uni = len(os&ns), len(os|ns)
                sim = inter/uni if uni else 0
                sims[prop], details[prop] = sim, {'type':'string','inter':list(os&ns),'union':list(os|ns)}
    return sims, details

# 속성별 유사도 팝업
def show_property_similarity():
    orig, new = txt_orig.get('1.0',tk.END), txt_new.get('1.0',tk.END)
    orig_vec, new_vec = vectorize_html(orig), vectorize_html(new)
    sims, det = calculate_property_similarity(orig_vec, new_vec)
    win = tk.Toplevel(root); win.title("속성별 유사도"); win.geometry("800x600")
    lframe,rframe = ttk.Frame(win), ttk.Frame(win)
    lframe.pack(side='left',fill='both',expand=True); rframe.pack(side='right',fill='both',expand=True)
    tree = ttk.Treeview(lframe,columns=("속성","유사도"),show='headings')
    tree.heading("속성",text="속성"); tree.heading("유사도",text="유사도")
    sb=ttk.Scrollbar(lframe,orient='vertical',command=tree.yview); tree.configure(yscrollcommand=sb.set)
    detail = scrolledtext.ScrolledText(rframe,width=50,height=30)
    detail.pack(fill='both',expand=True)
    for p,s in sorted(sims.items(),key=lambda x:-x[1]): tree.insert('','end',values=(p,f"{s:.4f}"))
    def on_sel(e):
        sel=tree.selection()
        if not sel: return
        prop=tree.item(sel[0])['values'][0]; info=det[prop]
        detail.delete('1.0',tk.END)
        for k,v in info.items(): detail.insert(tk.END,f"{k}: {v}\n")
    tree.bind('<<TreeviewSelect>>',on_sel)
    tree.pack(side='left',fill='both',expand=True); sb.pack(side='right',fill='y')

# 태그별 부합도 팝업
def show_tag_similarity():
    orig,new=txt_orig.get('1.0',tk.END),txt_new.get('1.0',tk.END)
    orig_vec,new_vec=vectorize_html(orig),vectorize_html(new)
    dv,of,labels=prepare_classifier(orig_vec)
    nf=dv.transform([v['style_vector'] for v in new_vec])
    scores=calculate_conformity_knn(of,nf,5)
    win=tk.Toplevel(root); win.title("태그별 부합도"); win.geometry("900x600")
    lf,rf=ttk.Frame(win),ttk.Frame(win)
    lf.pack(side='left',fill='both',expand=True); rf.pack(side='right',fill='both',expand=True)
    tree=ttk.Treeview(lf,columns=("태그","부합도"),show='headings')
    tree.heading("태그",text="태그 (HTML)"); tree.heading("부합도",text="부합도")
    sb=ttk.Scrollbar(lf,orient='vertical',command=tree.yview); tree.configure(yscrollcommand=sb.set)
    detail=scrolledtext.ScrolledText(rf,width=50,height=30)
    detail.pack(fill='both',expand=True)
    mapping={}
    for i,(vec,r) in enumerate(zip(new_vec,new_vec)):
        html=r['tag']; sc=scores[i]
        tree.insert('','end',values=(html,f"{sc:.4f}"))
        mapping[html]=(r['style_vector'],sc)
    def on_sel(e):
        s=tree.selection()
        if not s: return
        html,sc_str=tree.item(s[0])['values']; vec,sc=mapping[html]
        detail.delete('1.0',tk.END)
        detail.insert(tk.END,f"태그:\n{html}\n\n부합도: {sc:.4f}\n\n스타일 벡터:\n")
        for k,v in vec.items(): detail.insert(tk.END,f"  {k}: {'{:.2f}'.format(v) if isinstance(v,float) else v}\n")
    tree.bind('<<TreeviewSelect>>',on_sel)
    tree.pack(side='left',fill='both',expand=True); sb.pack(side='right',fill='y')

# 파일 로드
def load_file():
    try:
        with open('existed_html.txt','r',encoding='utf-8') as f:
            txt_orig.delete('1.0',tk.END); txt_orig.insert(tk.END,f.read())
    except Exception as e:
        messagebox.showerror("오류",f"원본 읽기 오류: {e}")
    try:
        with open('valid_generated_html.txt','r',encoding='utf-8') as f:
            txt_new.delete('1.0',tk.END); txt_new.insert(tk.END,f.read())
    except Exception as e:
        messagebox.showerror("오류",f"생성 읽기 오류: {e}")

# PCA 시각화
def plot_vectors(orig_feats, labels, new_feats, new_labels, overall_similarity):
    pca = PCA(n_components=2)
    all_feats = np.vstack([orig_feats, new_feats])
    reduced = pca.fit_transform(all_feats)
    n_o = len(orig_feats)
    o_red, n_red = reduced[:n_o], reduced[n_o:]
    plt.figure(figsize=(10,8))
    for lab in ['heading','body']:
        mask=[l==lab for l in labels]
        plt.scatter(o_red[mask,0],o_red[mask,1],label=f"Original {lab}",alpha=0.6)
        mask2=[l==lab for l in new_labels]
        plt.scatter(n_red[mask2,0],n_red[mask2,1],marker='^',label=f"New {lab}",alpha=0.6)
    plt.title(f"Style Vector PCA Visualization (Overall Similarity: {overall_similarity:.4f})")
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.grid(True,alpha=0.3); plt.show()

# 보고서 옵션 UI
def get_report_options():
    popup=tk.Toplevel(root); popup.title("보고서 옵션"); popup.geometry("300x300")
    frame=ttk.Frame(popup,padding="10"); frame.pack(fill='both',expand=True)
    opt=tk.StringVar(value="project")
    ttk.Label(frame,text="보고서 유형 선택:").pack(pady=(0,10))
    for v in ['project','comparison']:
        ttk.Radiobutton(frame,text=v,variable=opt,value=v).pack(anchor='w',pady=2)
    ttk.Label(frame,text="파일 이름 입력:").pack(pady=(10,5))
    fn=tk.StringVar(); ent=ttk.Entry(frame,textvariable=fn,width=30); ent.pack(pady=5)
    result={'filename':None,'option':None}
    def on_ok():
        name=fn.get().strip()
        if name:
            result['filename'],result['option']=name,opt.get(); popup.destroy()
    def on_cancel(): popup.destroy()
    btnf=ttk.Frame(frame); btnf.pack(pady=5)
    ttk.Button(btnf,text="확인",command=on_ok).pack(side='left',padx=5)
    ttk.Button(btnf,text="취소",command=on_cancel).pack(side='left',padx=5)
    ent.bind('<Return>',lambda e:on_ok())
    popup.transient(root); popup.grab_set(); root.wait_window(popup)
    return result['filename'], result['option']

# JSON 보고서 생성
def generate_report():
    orig, new = txt_orig.get('1.0',tk.END), txt_new.get('1.0',tk.END)
    if not os.path.exists('data'): os.makedirs('data')
    orig_vec, new_vec = vectorize_html(orig), vectorize_html(new)
    dv, of, labels = prepare_classifier(orig_vec)
    nf = dv.transform([v['style_vector'] for v in new_vec])
    scaler = StandardScaler()
    names = dv.get_feature_names_out()
    idx=[i for i,f in enumerate(names) if f in NUMERIC_PROPS]
    if idx: of[:,idx]=scaler.fit_transform(of[:,idx]); nf[:,idx]=scaler.transform(nf[:,idx])
    scores = calculate_conformity_knn(of, nf, 5)
    overall = float(np.mean(scores))
    prop_s, prop_d = calculate_property_similarity(orig_vec,new_vec)
    tag_s = {v['tag']:{'similarity':float(scores[i]),'style_vector':v['style_vector']} for i,v in enumerate(new_vec)}
    report = {'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'overall_similarity':overall,
              'property_similarities':prop_s,
              'property_details':prop_d,
              'tag_similarities':tag_s}
    fn,opt = get_report_options()
    if fn:
        if not fn.endswith('.json'): fn+='.json'
        folder=os.path.join('data',opt)
        if not os.path.exists(folder): os.makedirs(folder)
        path=os.path.join(folder,fn)
        with open(path,'w',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
        messagebox.showinfo("보고서 생성",f"보고서가 {path}에 저장되었습니다.")

# GUI 초기화 및 배치
root=tk.Tk(); root.title("HTML 인라인 스타일 벡터화 및 k-NN 부합도 시각화")
tk.Label(root,text="원본 HTML").grid(row=0,column=0,padx=5,pady=5)
txt_orig=scrolledtext.ScrolledText(root,width=50,height=15); txt_orig.grid(row=1,column=0,padx=5,pady=5)
tk.Label(root,text="생성 HTML").grid(row=0,column=1,padx=5,pady=5)
txt_new=scrolledtext.ScrolledText(root,width=50,height=15); txt_new.grid(row=1,column=1,padx=5,pady=5)
btn_load=tk.Button(root,text="파일 로드",command=load_file); btn_load.grid(row=2,column=0,columnspan=2,pady=5)
btn_run=tk.Button(root,text="비교·시각화 시작",command=compare_html); btn_run.grid(row=3,column=0,columnspan=2,pady=5)
btn_prop=tk.Button(root,text="속성별 유사도 확인",command=show_property_similarity); btn_prop.grid(row=4,column=0,columnspan=2,pady=5)
btn_tag=tk.Button(root,text="태그별 부합도 확인",command=show_tag_similarity); btn_tag.grid(row=5,column=0,columnspan=2,pady=5)
btn_report=tk.Button(root,text="보고서 생성",command=generate_report); btn_report.grid(row=6,column=0,columnspan=2,pady=5)
tk.Label(root,text="스타일 속성 처리 로그").grid(row=7,column=0,columnspan=2,padx=5,pady=5)
txt_log=scrolledtext.ScrolledText(root,width=100,height=10,state='disabled'); txt_log.grid(row=8,column=0,columnspan=2,padx=5,pady=5)
tk.Label(root,text="원본 벡터화 결과").grid(row=9,column=0,padx=5,pady=5)
txt_orig_result=scrolledtext.ScrolledText(root,width=50,height=20,state='disabled'); txt_orig_result.grid(row=10,column=0,padx=5,pady=5)
tk.Label(root,text="변경된 벡터화 결과").grid(row=9,column=1,padx=5,pady=5)
txt_new_result=scrolledtext.ScrolledText(root,width=50,height=20,state='disabled'); txt_new_result.grid(row=10,column=1,padx=5,pady=5)
root.after(100, load_file)
root.mainloop()
