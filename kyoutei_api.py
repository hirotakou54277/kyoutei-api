# kyoutei_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn
import joblib
from itertools import combinations, permutations
from bs4 import BeautifulSoup
import requests
import re
import time

app = FastAPI()

base_features = ["勝率", "複勝率", "平均スタートタイミング", "今期能力指数"]
course_features = []
for i in range(1, 7):
    course_features += [
        f"{i}コース平均スタート順位", f"{i}コース複勝率",
        f"{i}コース1着回数", f"{i}コース2着回数", f"{i}コース3着回数"
    ]

class PairModel(nn.Module):
    def __init__(self, num_features, num_names, emb_dim=8):
        super().__init__()
        self.emb = nn.Embedding(num_names, emb_dim)
        self.fc1 = nn.Linear(num_features + emb_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 15)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, name):
        e = self.emb(name)
        x = torch.cat([x, e], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

class PredictRequest(BaseModel):
    place: str
    race: str

df_player_scaled = pd.read_csv("model_data/player_features.csv")
scaler = joblib.load("model_data/scaler.pkl")
name_encoder = joblib.load("model_data/name_encoder.pkl")

def clean_name(name):
    return re.sub(r'\s+', '', name)
df_player_scaled["名前漢字"] = df_player_scaled["名前漢字"].apply(clean_name)

def load_trained_model_auto(model_class, input_dim, model_path="model_data/model.pt"):
    state_dict = torch.load(model_path, map_location="cpu")
    emb_weight = state_dict["emb.weight"]
    name_count = emb_weight.shape[0]
    model = model_class(input_dim, name_count)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_trained_model_auto(PairModel, len(base_features + course_features))

place_dict = {'桐生':'01', '戸田':'02', '江戸川':'03','平和島':'04', '多摩川':'05', '浜名湖':'06','蒲郡':'07', '常滑':'08', '津':'09','三国':'10', 'びわこ':'11', '住之江':'12' ,'尼崎':'13', '鳴門':'14','丸亀':'15','児島':'16', '宮島':'17','徳山':'18', '下関':'19', '若松':'20','芦屋':'21', '福岡':'22', '唐津':'23', '大村':'24'}
race_dict = {str(i): f"{i:02d}" for i in range(1, 13)}

def extract_features(race_dict, df_player_scaled):
    features, ids = [], []
    for i in range(1, 7):
        name = race_dict[f'name_{i}']
        match = df_player_scaled[df_player_scaled["名前漢字"] == name]
        if match.empty:
            raise ValueError(f"選手 {name} の成績データが見つかりません")
        vec = match[base_features + course_features].values[0]
        name_id = match["名前漢字_encoded"].values[0]
        features.append(vec)
        ids.append(name_id)
    avg_feat = sum(features) / 6
    main_id = ids[0]
    return torch.tensor(avg_feat, dtype=torch.float32).unsqueeze(0), torch.tensor([main_id], dtype=torch.long), list(combinations([race_dict[f'name_{i}'] for i in range(1, 7)], 2))

@app.post("/predict")
def predict(req: PredictRequest):
    placeno = place_dict.get(req.place)
    raceno = race_dict.get(req.race)
    if not placeno or not raceno:
        raise HTTPException(status_code=400, detail="無効な会場名またはレース番号")

    today = time.strftime("%Y%m%d")
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={raceno}&jcd={placeno}&hd={today}"

    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
    wordclass1 = soup.find_all("div", {'class': 'is-fs18 is-fBold'})
    wordlist1 = [x.text.replace("\n", " ").replace("\r", " ").replace('\u3000', '').strip() for x in wordclass1]
    racer_names = [clean_name(name) for name in wordlist1[:6]]

    race_input = {f'name_{i+1}': racer_names[i] for i in range(6)}
    name_to_lane = {name: str(i + 1) for i, name in enumerate(racer_names)}
    X_new, name_new, all_pairs = extract_features(race_input, df_player_scaled)

    with torch.no_grad():
        probs = model(X_new, name_new).numpy()[0]

    ranked = np.argsort(probs)[::-1]
    result_lines = []
    count = 0
    lane_pool = set()

    result_lines.append("===== 拡連複 上位ペア予測（※1コース除外） =====")
    for idx in ranked:
        name1, name2 = all_pairs[idx]
        lane1 = name_to_lane.get(name1, "?")
        lane2 = name_to_lane.get(name2, "?")
        if lane1 == "1" or lane2 == "1":
            continue
        pair_str = f"{lane1}-{lane2}" if int(lane1) < int(lane2) else f"{lane2}-{lane1}"
        others = [lane1, lane2, "1"]
        patterns = [''.join(p) for p in permutations(others)]
        pattern_str = '  '.join(patterns)
        result_lines.append(f"{pair_str} : 確率 {probs[idx]:.3f}　　1コースを含む組み合わせ({pattern_str})")
        lane_pool.update([lane1, lane2])
        count += 1
        if count >= 5:
            break

    top_lanes = sorted(lane_pool)
    if len(top_lanes) >= 4:
        selected = top_lanes[:4]
        perms_3 = list(permutations(selected, 3))
        result_lines.append(f"\n確率上位3行の数字は {''.join(selected)} です。\n")
        result_lines.append(f"{len(perms_3)}通りの1コースを省いた3連組み合わせ（順列）:")
        for i in range(0, len(perms_3), 6):
            row = [''.join(p) for p in perms_3[i:i+6]]
            result_lines.append('  '.join(row))
    else:
        result_lines.append("⚠️ 上位ペアから十分なコース番号（4種）が取得できませんでした。")

    return {"message": "\n".join(result_lines)}
