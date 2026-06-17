import pandas as pd
import json
import os
import pickle
import argparse
import os
from tqdm import tqdm
import pandas as pd
import warnings
from datetime import timedelta
import json
import ast

warnings.filterwarnings("ignore")


reverse_map = {
    "no change": "no change",
    "worsened": "improved",
    "improved": "worsened",
}
def replace_ext_to_jpg(path):
    base = os.path.splitext(path)[0]
    return base + ".jpg"

def readJSON(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
            return data
    except Exception as e:
        print('File does not exist',filepath)
        return None

def get_disease_progression_labels(scene_graph,save_path):
    # /silver_dataset/scene_graph/scene_graph/0a0a1d17-5fb4ffcc-a8ca1541-8c44d583-fd9c6388_SceneGraph.json
    tab_dict={'subject_id':[],'study_id':[],'cxr1_dicom_id':[],'cxr2_dicom_id':[],'disease_progression':[]}
    graph_files =[i for i in os.listdir(scene_graph) if i.endswith('_SceneGraph.json')]
    # graph_files=graph_files[:2] 
    for graph in tqdm(graph_files,total=len(graph_files),desc='get disease progression labels from scene graph'):
        data=readJSON(os.path.join(scene_graph,graph))
        # breakpoint()
        comparisons = data['relationships']
        if len(comparisons)==0:
            continue
        tab_dict['subject_id'].append(data['patient_id'])
        tab_dict['study_id'].append(data['study_id']) 

        # The 'relationship_id' uniquely identifies each comparison relationship between the object ('subject_id') on the current exam and the object ('object_id' for the same anatomical location) from the previous exam. 
        tab_dict['cxr2_dicom_id'].append(comparisons[0]['subject_id'].split('_')[0])# 这是当前的内容
        tab_dict['cxr1_dicom_id'].append(comparisons[0]['object_id'].split('_')[0]) # 这是之前的内容
        disease_progression=[]
        for comp in comparisons:
            compare = comp['relationship_names']
            compare = [x for x in compare if x in ['comparison|yes|worsened'
                                                   ,'comparison|yes|improved','comparison|yes|no change']]
            # compare = ';;'.join(sorted([x.split('|')[2] for x in compare]))
            compare =[x.split('|')[2] for x in compare]
            if len(compare)>0:
                # get turple (bbox_name,[attributes],[compare])
                turple = [comp['bbox_name'],[i.split("|")[-1] for i in comp['attributes'] if i.split('|')[0] in ['anatomicalfinding','disease']],compare,comp.get('phrase', '')]
                disease_progression.append(turple)
            
        tab_dict['disease_progression'].append(disease_progression)
    # breakpoint()
    tab=pd.DataFrame(tab_dict)
    tab.to_csv(os.path.join(save_path,'all_comparison.csv'),index=False)

def get_disease_and_progression(row):
    disease_progression = row['disease_progression']
    # 只保留 progression 唯一的关系
    disease_progression = [i for i in disease_progression if len(i[2]) == 1]

    # entity -> {'progressions': [...], 'phrases': [...], 'bboxes': [...], 'is_disease': [...]}
    # entity 可以是 disease 名，也可以是 bbox_name（当 diseases 为空时）
    entity_dict = {}
    for organ in disease_progression:
        bbox_name, diseases, compares, phrase = organ  # i[0], i[1], i[2], i[3]
        prog = compares[0]  # 'worsened' / 'improved' / 'no change'

        if len(diseases) > 0:
            # 有疾病标签：这些 entity 是疾病
            targets = diseases
            is_disease_flag = 1
        else:
            # 没有疾病标签：用 bbox_name 本身作为 entity，标记为非疾病
            targets = [bbox_name]
            is_disease_flag = 0

        for ent in targets:
            if ent not in entity_dict:
                entity_dict[ent] = {
                    'progressions': [],
                    'phrases': [],
                    'bboxes': [],
                    'is_disease': []
                }
            entity_dict[ent]['progressions'].append(prog)
            entity_dict[ent]['phrases'].append(phrase)
            entity_dict[ent]['bboxes'].append(bbox_name)
            entity_dict[ent]['is_disease'].append(is_disease_flag)

    # 过滤掉 progression 有歧义的 entity，只保留 progression 一致的
    final_dict = {}
    for ent, info in entity_dict.items():
        if len(set(info['progressions'])) == 1:
            progression = info['progressions'][0]

            # 去重合并 phrase，用逗号连接
            phrases = list({
                p.strip()
                for p in info['phrases']
                if isinstance(p, str) and p.strip() != ''
            })
            combined_phrase = ','.join(phrases) if len(phrases) > 0 else ''

            # 去重合并 bbox_name，用逗号连接
            bboxes = list({
                b.strip()
                for b in info['bboxes']
                if isinstance(b, str) and b.strip() != ''
            })
            combined_bbox = ','.join(bboxes) if len(bboxes) > 0 else ''

            # is_disease：如果这个 entity 的标记一致，就取该值；否则就设 0
            if len(set(info['is_disease'])) == 1:
                is_disease = info['is_disease'][0]
            else:
                is_disease = 0

            # ent 可以是疾病，也可以是解剖名称
            # 返回: (progression, phrase, bbox_name, is_disease)
            final_dict[ent] = (progression, combined_phrase, combined_bbox, is_disease)

    return final_dict

def get_neighbor_cxr_labels(all_comparison,save_path):
    
    neighbor_progression_dict={'subject_id':[],'study_id':[],'cxr1_dicom_id':[],'cxr2_dicom_id':[],'disease':[],'progression':[], 'phrase':[],'bbox_name': [],'is_disease': [] }
    all_comparison=pd.read_csv(all_comparison)
    # turn the string to list
    all_comparison['disease_progression']=all_comparison['disease_progression'].apply(lambda x: ast.literal_eval(x))
    
    # get disease and their progression labels
    all_comparison['disease_dict'] = all_comparison.apply(get_disease_and_progression,axis=1)
    
    # drop the rows with empty disease_dict
    all_comparison = all_comparison[all_comparison['disease_dict'].apply(lambda x: len(x)>0)]
    
    # get the disease and progression labels
    for _,row in tqdm(all_comparison.iterrows(),total=len(all_comparison),desc='get progression labels for CXR pairs'):
        disease_dict = row['disease_dict']
        for disease, (progression, phrase, bbox_name, is_disease) in disease_dict.items():
            neighbor_progression_dict['subject_id'].append(row['subject_id'])
            neighbor_progression_dict['study_id'].append(row['study_id'])
            neighbor_progression_dict['cxr1_dicom_id'].append(row['cxr1_dicom_id'])
            neighbor_progression_dict['cxr2_dicom_id'].append(row['cxr2_dicom_id'])
            neighbor_progression_dict['disease'].append(disease)
            neighbor_progression_dict['progression'].append(progression)
            neighbor_progression_dict['phrase'].append(phrase)
            neighbor_progression_dict['bbox_name'].append(bbox_name) 
            neighbor_progression_dict['is_disease'].append(is_disease) 

    neighbor_progression_df=pd.DataFrame(neighbor_progression_dict)
    neighbor_progression_df.to_csv(os.path.join(save_path,'neighbor_progression.csv'),index=False)

def correct_using_golden_labels(golden_dir,neighbor_progression):
    # correct and also using the golden labels to evaluate
    # get [patient_id,current_image_id,previous_image_id,label_name,comparison] from the golden labels
    golden_labels=pd.read_csv(golden_dir, sep='\t')[['patient_id','current_image_id','previous_image_id','label_name','comparison']]
    # rename to ['subject_id','cxr2_dicom_id','cxr1_dicom_id','disease','progression']
    golden_labels.columns=['subject_id','cxr2_dicom_id','cxr1_dicom_id','disease','progression']
    # drop dupilcates
    golden_labels=golden_labels.drop_duplicates()
    golden_cxr_pairs = set(golden_labels[['cxr2_dicom_id', 'cxr1_dicom_id']].apply(tuple, axis=1))

    neighbor_progression_origin=pd.read_csv(neighbor_progression)
    neighbor_progression=neighbor_progression_origin[neighbor_progression_origin[['cxr2_dicom_id','cxr1_dicom_id']].apply(lambda x: tuple(x) in golden_cxr_pairs,axis=1)]
    # merge the golden_labels with neighbor_progression
    neighbor_progression_with_golden=neighbor_progression.merge(golden_labels,on=['subject_id','cxr1_dicom_id','cxr2_dicom_id','disease'],how='outer',suffixes=('_neighbor', '_golden'))
    # fill na with 'empty'
    neighbor_progression_with_golden=neighbor_progression_with_golden.fillna('empty')

    
    # correct the neighbor_progression using golden_labels
    neighbor_progression_origin['corrected_progression']=neighbor_progression_origin.apply(lambda x: _get_golden_labels(x,golden_labels,golden_cxr_pairs),axis=1)
    # drop empty in the corrected_progression
    neighbor_progression_origin=neighbor_progression_origin[neighbor_progression_origin['corrected_progression']!='empty']
    # mark rows from golden_labels
    neighbor_progression_origin['from_golden']=neighbor_progression_origin[['cxr2_dicom_id','cxr1_dicom_id']].apply(lambda x: 1 if tuple(x) in golden_cxr_pairs else 0,axis=1)
    assert 'empty' not in neighbor_progression_origin['corrected_progression'].values
    neighbor_progression_origin.to_csv(os.path.join(args.save_path,'golden_corrected_neighbor_progression.csv'),index=False)


def _get_golden_labels(row,golden_df,golden_cxr_pairs):
    cxr2_dicom_id = row['cxr2_dicom_id']
    cxr1_dicom_id = row['cxr1_dicom_id']
    if (cxr2_dicom_id, cxr1_dicom_id) not in golden_cxr_pairs:
        return row['progression']
    # get the golden labels
    golden_labels=golden_df[(golden_df['cxr2_dicom_id']==cxr2_dicom_id)&(golden_df['cxr1_dicom_id']==cxr1_dicom_id)]
    if len(golden_labels)==0:
        return 'empty'
    else:
        return golden_labels['progression'].values[0]


def pair_progression_statistic(neighbor_progression,cxr_stay,save_dir):
    neighbor_progression=pd.read_csv(neighbor_progression)

    # disease, progression, count
    print(neighbor_progression.groupby(['disease','progression']).size().reset_index(name='count'))
    # # disease, count
    print(neighbor_progression.groupby(['disease']).size().reset_index(name='count'))

    cxr_stay=pd.read_csv(cxr_stay)
    cxr_stay_sorted = cxr_stay.sort_values(by=['subject_id', 'stay_id', 'CXRRelTime'])

    image_pairs = []

    for (subject_id, stay_id), group in cxr_stay_sorted.groupby(['subject_id', 'stay_id']):
        dicom_ids = group['dicom_id'].tolist()
        
        # generate image pairs
        for i in range(len(dicom_ids) - 1):
            image_pairs.append({
                'subject_id': subject_id,
                'stay_id': stay_id,
                'cxr1_dicom_id': dicom_ids[i],
                'cxr2_dicom_id': dicom_ids[i + 1],
                'cxr1_time': group.iloc[i]['CXRRelTime'],
                'cxr2_time': group.iloc[i + 1]['CXRRelTime']
            })  
    image_pairs_in_stay = pd.DataFrame(image_pairs)
    # neighbor pairs should be in the image_pairs
    image_pairs_in_stay_dicom_id = set(image_pairs_in_stay[['cxr2_dicom_id', 'cxr1_dicom_id']].apply(tuple, axis=1))
    neighbor_progression=neighbor_progression[neighbor_progression[['cxr2_dicom_id','cxr1_dicom_id']].apply(lambda x: tuple(x) in image_pairs_in_stay_dicom_id,axis=1)]
    

    # get image pairs and progression for each disease
    disease_list=['lung opacity','pleural effusion','atelectasis','enlarged cardiac silhouette','pulmonary edema/hazy opacity','consolidation','pneumonia']
    # merge the neighbor_progression with image_pairs_in_stay
    neighbor_progression=neighbor_progression.merge(image_pairs_in_stay,on=['subject_id','cxr1_dicom_id','cxr2_dicom_id'],how='left')
    save_path=os.path.join(save_dir,'disease_progressions')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for disease in disease_list:
        # get the image pairs for each disease
        disease_image_pairs=neighbor_progression[neighbor_progression['disease']==disease]
        disease_image_pairs.to_csv(os.path.join(save_path,f'{disease.split("/")[0]}_image_pairs.csv'),index=False)

#  for reproducability (To get the same splits as the paper)
def apply_order_map(new_csv, order_map_path, keys=['subject_id','study_id','cxr1_dicom_id','cxr2_dicom_id']):
    new_df = pd.read_csv(new_csv)
    order_map = pd.read_csv(order_map_path)
    merged = new_df.merge(order_map, on=keys, how='left')
    merged = merged.sort_values(by='order_index').drop(columns=['order_index'])
    merged.to_csv(new_csv, index=False)



def main():
    pairs_csv_path = "./dataset/golden_corrected_neighbor_progression.csv"   
    meta_csv_path = "./dataset/utils/cxr-record-list_view.csv"            
    output_json_path = "./dataset/progression_data.json" 
    image_root = "./dataset" 
    df_pairs = pd.read_csv(pairs_csv_path)  
    df_meta = pd.read_csv(meta_csv_path)     

    meta_index = {}
    for _, row in df_meta.iterrows():
        subj = int(row["subject_id"])
        dicom = str(row["dicom_id"])
        study = int(row["study_id"])
        path = str(row["path"])
        meta_index[(subj, dicom)] = (study, path)

    group_cols = ["subject_id", "study_id", "cxr1_dicom_id", "cxr2_dicom_id"]
    grouped = df_pairs.groupby(group_cols, dropna=False)

    samples = []

    raw_pair_count = 0
    skipped_no_progression = 0
    skipped_missing_img = 0
    missing_meta_ctx = 0
    missing_meta_cur = 0

    for (subject_id, study_id_pairs, cxr1_id, cxr2_id), group in grouped:
        raw_pair_count += 1
        subject_id_int = int(subject_id)
        cxr1_id_str = str(cxr1_id)
        cxr2_id_str = str(cxr2_id)

        # --------------------------------------------------
        # 1) 收集去重后的 (entity, progression, bbox_name, is_disease)
        # --------------------------------------------------
        pairs = set()
        for _, row in group.iterrows():
            ent = str(row["disease"]).strip()  # 对疾病或非疾病实体，统一叫 entity
            corr_prog = str(row["corrected_progression"]).strip()
            bbox_name = str(row.get("bbox_name", "")).strip()
            # is_disease 可能是 float，先处理 NaN
            is_dis_val = row.get("is_disease", 1)
            if pd.isna(is_dis_val):
                is_dis = 1
            else:
                is_dis = int(is_dis_val)

            pairs.add((ent, corr_prog, bbox_name, is_dis))

        progression_items = []
        reversed_items = []
        entity_items = []

        # --------------------------------------------------
        # 2) 按 entity 构造 progression_items / reversed_items
        #    - 疾病: "disease location of bbox progression"
        #    - 非疾病: "entity progression"
        # --------------------------------------------------
        for ent, corr_prog, bbox_name, is_dis in pairs:
            if corr_prog == "" or str(corr_prog).lower() == "nan":
                continue

            corr_prog_clean = str(corr_prog).strip()
            corr_prog_lower = corr_prog_clean.lower()
            rev_label = reverse_map.get(corr_prog_lower, corr_prog_clean)

            if is_dis == 1:
                # 疾病实体
                if bbox_name:
                    prog_piece = f"{ent} at {bbox_name} {corr_prog_clean}"
                    rev_piece = f"{ent} at {bbox_name} {rev_label}"
                else:
                    prog_piece = f"{ent} {corr_prog_clean}"
                    rev_piece = f"{ent} {rev_label}"
                #prog_piece = f"{ent} {corr_prog_clean}"
                #rev_piece = f"{ent} {rev_label}"
            #else:  这里使生成的json文件不包含解剖结构！！
                # 非疾病实体（解剖 / 其他）
            #    prog_piece = f"{ent} {corr_prog_clean}"
            #    rev_piece = f"{ent} {rev_label}"

            progression_items.append(prog_piece)
            reversed_items.append(rev_piece)
            entity_items.append(ent)

        if len(progression_items) == 0:
            skipped_no_progression += 1
            continue

        progression_str = ". ".join(progression_items)
        reversed_progression_str = ". ".join(reversed_items)
        entity_str = ". ".join(entity_items)

        if progression_str == reversed_progression_str:
            no_change_mask = 1
        else:
            no_change_mask = 0

        # --------------------------------------------------
        # 3) 找到 context / current 图像路径
        # --------------------------------------------------
        ctx_key = (subject_id_int, cxr1_id_str)  # context image
        if ctx_key in meta_index:
            ctx_study_id_meta, ctx_path_meta = meta_index[ctx_key]
            ctx_rel_path = replace_ext_to_jpg(ctx_path_meta)
        else:
            missing_meta_ctx += 1
            # 找不到 meta 就跳过这一对
            print(f"[Missing meta] context image for subject={subject_id_int}, dicom={cxr1_id_str}")
            continue

        cur_key = (subject_id_int, cxr2_id_str)  # current image
        if cur_key in meta_index:
            cur_study_id_meta, cur_path_meta = meta_index[cur_key]
            img_rel_path = replace_ext_to_jpg(cur_path_meta)
        else:
            missing_meta_cur += 1
            print(f"[Missing meta] current image for subject={subject_id_int}, dicom={cxr2_id_str}")
            continue

        img_abs = os.path.join(image_root, img_rel_path)
        ctx_abs = os.path.join(image_root, ctx_rel_path)

        if (not os.path.exists(img_abs)) or (not os.path.exists(ctx_abs)):
            skipped_missing_img += 1
            print(f"[Missing] img: {img_abs} exists={os.path.exists(img_abs)}, "
                   f"ctx: {ctx_abs} exists={os.path.exists(ctx_abs)}")
            continue

    
        sample = {
            "id": cxr2_id_str,                # 以当前 CXR 的 dicom_id 作为样本 id
            "study_id": int(study_id_pairs), 
            "subject_id": subject_id_int,
            "image_path": [img_rel_path],     # 当前图
            "context_image": [ctx_rel_path],  # 前一张图
            "progression": progression_str,               # disease / entity progression 描述
            "reversed_progression": reversed_progression_str,  # 反向 progression
            "entity": entity_str,
            "no_change_mask": no_change_mask,
        }
        samples.append(sample)


    with open(output_json_path, "w") as f:
        json.dump(samples, f, indent=2)

    # 6) 打印统计信息
    print(f"Raw pairs (grouped)                    : {raw_pair_count}")
    print(f"Valid samples (after all filtering)    : {len(samples)}")
    print(f"Skipped (no valid progression)         : {skipped_no_progression}")
    print(f"Skipped (missing image files)          : {skipped_missing_img}")
    print(f"Missing meta for context images (cxr1) : {missing_meta_ctx}")
    print(f"Missing meta for current images (cxr2) : {missing_meta_cur}")


def merge_split(samples, path2sent, split_name=""):
    MIMIC_CXR_DATA_DIR = "/bulk/zhangj41/zion/JJ_Peoton/MultiModal/mimic-cxr-jpg/2.0.0/"
    """
    处理一个 split（train / val / test）：
      - 为每个样本找 current / prior 的 captions
      - 找不到就丢弃
      - 找到就写入新字段
    """
    kept = 0
    missing_caption = 0
    new_samples = []

    for s in tqdm(samples, desc=f"Merging {split_name}"):
        # 取出当前 & 历史图像路径（json 中是 list，这里取第一个）
        cur_rel_list = s.get("image_path", [])
        prior_rel_list = s.get("context_image", [])
        if not cur_rel_list or not prior_rel_list:
            missing_caption += 1
            continue

        cur_rel = cur_rel_list[0]
        prior_rel = prior_rel_list[0]

        cur_abs = os.path.join(MIMIC_CXR_DATA_DIR, cur_rel)
        prior_abs = os.path.join(MIMIC_CXR_DATA_DIR, prior_rel)
        print(cur_abs, prior_abs)

        cur_caps = path2sent.get(cur_abs, None)
        prior_caps = path2sent.get(prior_abs, None)

        # 任意一边没有 caption，就丢弃
        if not cur_caps or not prior_caps:
            print('miss')
            missing_caption += 1
            continue

        # 构建新样本
        new_s = dict(s)
        new_s["current_full_path"] = cur_abs
        new_s["prior_full_path"] = prior_abs
        new_s["current_captions"] = cur_caps             # 句子列表
        new_s["prior_captions"] = prior_caps             # 句子列表
        new_s["current_caption"] = ". ".join(cur_caps)     # 拼成一整段
        new_s["prior_caption"] = ". ".join(prior_caps)

        new_samples.append(new_s)
        kept += 1

    print(f"\n[{split_name}] total={len(samples)}, kept={kept}, dropped(no caption)={missing_caption}\n")
    return new_samples


def mainv2():
    INPUT_JSON = "./dataset/progression_data.json"
    CAPTION_PICKLE = "./dataset/captions.pickle"  # from mimic cxr jpg
    OUTPUT_JSON = "./dataset/progression_with_captions.json"
    # 1. 读 captions.pickle
    with open(CAPTION_PICKLE, "rb") as f:
        path2sent = pickle.load(f)
    print(f"Loaded {len(path2sent)} caption entries from {CAPTION_PICKLE}")
    print(path2sent)

    # 2. 读原始 json
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. 处理结构：支持 dict(train/val/test) 或 list
    if isinstance(data, dict) and any(k in data for k in ["train", "val", "test"]):
        new_data = {}
        for split in data:
            new_data[split] = merge_split(data[split], path2sent, split_name=split)
    elif isinstance(data, list):
        # 顶层直接是 list 的情况
        new_data = merge_split(data, path2sent, split_name="all")
    else:
        raise ValueError("Unsupported JSON structure, expect dict with train/val/test or a list.")

    # 4. 写出新 json
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved merged json with captions to: {OUTPUT_JSON}")

def dicom_id_from_path(p: str) -> str:
    """Extract dicom_id from a path like .../<dicom_id>.jpg"""
    if p is None:
        return ""
    p = str(p).strip()
    if p == "":
        return ""
    base = os.path.basename(p)
    if base.endswith(".jpg"):
        base = base[:-4]
    return base.strip()

def load_json_list(json_path):
    """
    Supports:
      - top-level list
      - top-level dict with a list under list_key or common keys
    Returns: (records, wrapper_dict_or_None, used_key_or_None)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        return obj['test']

def build_test_pairs_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    pairs = set()
    for _, row in df.iterrows():
        cur = dicom_id_from_path(row["dicom_id"])          
        pri = dicom_id_from_path(row["previous_dicom_id"])  
        pairs.add((cur, pri))
        print('csv pairs:', (cur, pri))
    return pairs


def build_test_pairs_from_split_json(split_json_path):
    records = load_json_list(split_json_path)

    pairs = set()
    kept_test_items = 0

    for item in records:
        if not isinstance(item, dict):
            continue
        if str(item.get("split", "")).strip().lower() != "test":
            continue

        cur = str(item.get("id", "")).strip()
        ctx = item.get("context_image", None)
        pri = ""
        if isinstance(ctx, list) and len(ctx) > 0:
            pri = dicom_id_from_path(ctx[0])
        elif isinstance(ctx, str):
            pri = dicom_id_from_path(ctx)

        if cur and pri:
            print('split json pairs:', (cur, pri))
            pairs.add((cur, pri))
            kept_test_items += 1

    return pairs, kept_test_items

def get_pretrain_pair(sample: dict):
    """
    pretrain sample example:
      - id: current dicom_id
      - prior_full_path: ".../<prior_id>.jpg" (preferred)
      - or context_image: [".../<prior_id>.jpg"]
    """
    cur = str(sample.get("id", "")).strip()

    pri = ""
    if "prior_full_path" in sample:
        pri = dicom_id_from_path(sample.get("prior_full_path"))
    if not pri:
        ctx = sample.get("context_image", None)
        if isinstance(ctx, list) and len(ctx) > 0:
            pri = dicom_id_from_path(ctx[0])
        elif isinstance(ctx, str):
            pri = dicom_id_from_path(ctx)

    return cur, pri

def mainv3():
    pretrain_json ='./dataset/progression_with_captions.json'
    test_csv = './MS_CXR_T_temporal_image_classification_v1.0.0.csv'  # exclude the leaked pairs in this csv from the pretrain samples
    out_json='./dataset/progression_final.json'

    leak_pairs_csv = build_test_pairs_from_csv(test_csv)
    print(f"[Info] TEST CSV pairs: {len(leak_pairs_csv)}")


    pretrain_records = load_json_list(pretrain_json)
    print(f"[Info] Pretrain samples loaded: {len(pretrain_records)}")

    kept = []
    leaked_list = []
    missing = 0

    for s in pretrain_records:
        if not isinstance(s, dict):
            kept.append(s)
            continue

        cur, pri = get_pretrain_pair(s)
        if not cur or not pri:
            missing += 1
            kept.append(s)
            continue
        print((cur, pri))
        if (cur, pri) in leak_pairs_csv:
            leaked_list.append({"id": cur, "prior_id": pri})
        else:
            kept.append(s)

    print(f"[Result] Leaked removed: {len(leaked_list)}")
    print(f"[Result] Kept: {len(kept)}")
    if missing:
        print(f"[Warn] Missing id/prior in pretrain samples (kept): {missing}")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Filtered pretrain JSON -> {out_json}")

if __name__ == '__main__':
    # 先从scene_graph中获取csv文件
    parser = argparse.ArgumentParser(description='get disease progression labels.')
    parser.add_argument('--scene_graph', type=str, default='your path for scene_graph', help='the all_stay_csv path, containing the all the icu_stay samples needed.')
    parser.add_argument('--golden_dir', type=str, default='your path for gold_object_comparison_with_coordinates.txt', help='the path of the golden labels')
    parser.add_argument('--save_path', type=str, default='your path for save files',  help='the path to save files')
    parser.add_argument('--order_map_path', type=str, default=os.path.join(os.path.dirname(__file__),'/all_comparison_order_map.csv'), help='the path of the all comparison order mapping for reproducebility')
    
    args, _ = parser.parse_known_args()
    '''get_disease_progression_labels(args.scene_graph, args.save_path)
    all_comparison_path=os.path.join(args.save_path,'all_comparison.csv')
    #apply_order_map(all_comparison_path, args.order_map_path)
    get_neighbor_cxr_labels(all_comparison_path, args.save_path)
    correct_using_golden_labels(args.golden_dir,os.path.join(args.save_path,'neighbor_progression.csv'))'''

    
    # 再将csv文件整理成可以使用的json文本信息
    main()
    mainv2()
    mainv3()
    # ===========================================
    
    
