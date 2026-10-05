"""Praveen SID 8511 Task 2: three from-scratch Yelp sentiment models.

No pretrained embeddings or language models are used. The script creates the
required evaluation tables, confidence intervals, McNemar tests, slices, and
20-case error-review worksheet. The default run uses the complete Yelp split;
--smoke-test is only for checking the pipeline.
"""
import argparse, json, random, re, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score,
    matthews_corrcoef, brier_score_loss)
from scipy.stats import binomtest

SID, SEED = 8511, 8511
STOP = set("a an the and or but if then than is are was were be been to of in on for with as by at from this that it its".split())
STOP -= {"no", "not", "never", "nor"}

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def tokenize(text):
    text = str(text).lower().replace("’", "'")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return [x for x in re.sub(r"\s+", " ", text).split() if x not in STOP]

def encode(rows, vocab, max_len):
    x = np.zeros((len(rows), max_len), dtype=np.int64)
    lengths = np.ones(len(rows), dtype=np.int64)
    y = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        ids = [vocab.get(t, 1) for t in tokenize(row["text"])][:max_len]
        if ids: x[i, :len(ids)] = ids; lengths[i] = len(ids)
        y[i] = int(row["label"])
    return x, y, lengths

class CNNClassifier(nn.Module):
    def __init__(self, v):
        super().__init__(); self.emb=nn.Embedding(v,128,padding_idx=0)
        self.conv=nn.Sequential(nn.Conv1d(128,128,5,padding=2),nn.ReLU(),nn.Dropout(.25),nn.Conv1d(128,128,3,padding=1),nn.ReLU())
        self.fc=nn.Linear(128,1)
    def forward(self,x,l): return self.fc(self.conv(self.emb(x).transpose(1,2)).amax(2)).squeeze(1)

class BiLSTMAttention(nn.Module):
    def __init__(self,v,hidden=128):
        super().__init__(); self.emb=nn.Embedding(v,160,padding_idx=0); self.rnn=nn.LSTM(160,hidden,batch_first=True,bidirectional=True)
        self.attn=nn.Linear(hidden*2,1); self.fc=nn.Sequential(nn.Dropout(.3),nn.Linear(hidden*2,1))
    def forward(self,x,l):
        h,_=self.rnn(self.emb(x)); mask=torch.arange(x.size(1),device=x.device)[None,:] >= l[:,None]
        a=self.attn(h).squeeze(-1).masked_fill(mask,-1e4).softmax(1); return self.fc((h*a.unsqueeze(-1)).sum(1)).squeeze(1)

class GRUMaxPool(nn.Module):
    def __init__(self,v):
        super().__init__(); self.emb=nn.Embedding(v,128,padding_idx=0); self.rnn=nn.GRU(128,96,batch_first=True,bidirectional=True); self.fc=nn.Sequential(nn.LayerNorm(192),nn.Dropout(.2),nn.Linear(192,1))
    def forward(self,x,l):
        h=self.rnn(self.emb(x))[0]; mask=(torch.arange(x.size(1),device=x.device)[None,:]>=l[:,None]).unsqueeze(-1)
        return self.fc(h.masked_fill(mask,-1e4).amax(1)).squeeze(1)

def ece(y,p,bins=15):
    out=0.; edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&((p<=hi) if hi==1 else (p<hi))
        if m.any(): out += m.mean()*abs(p[m].mean()-y[m].mean())
    return float(out)

def metrics(y,p):
    pred=(p>=.5).astype(int); d={"accuracy":accuracy_score(y,pred)}
    for a in ("macro","micro","weighted"):
        pr,re,f,_=precision_recall_fscore_support(y,pred,average=a,zero_division=0)
        d.update({f"precision_{a}":pr,f"recall_{a}":re,f"f1_{a}":f})
    d.update(roc_auc=roc_auc_score(y,p),pr_auc=average_precision_score(y,p),
             mcc=matthews_corrcoef(y,pred),brier_score=brier_score_loss(y,p),
             ece_15_bins=ece(y,p))
    return d, pred

def train_one(model, name, loaders, device, epochs, out):
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.999),weight_decay=1e-4)
    loss_fn=nn.BCEWithLogitsLoss(); best=(-1,None); hist=[]; t=time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for ep in range(1,epochs+1):
        model.train(); seen=0; total=0.
        for x,y,l in loaders[0]:
            x,y,l=x.to(device),y.float().to(device),l.to(device); opt.zero_grad()
            loss=loss_fn(model(x,l),y); loss.backward(); opt.step()
            total+=loss.item()*len(y); seen+=len(y)
        model.eval(); correct=0; n=0
        with torch.no_grad():
            for x,y,l in loaders[1]:
                z=model(x.to(device),l.to(device)); correct += ((z.sigmoid()>=.5).cpu()==y).sum().item(); n+=len(y)
        row={"model":name,"epoch":ep,"train_loss":total/seen,"val_accuracy":correct/n}; hist.append(row); print(row)
        with open(out.parent / "training_raw.log", "a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        if row["val_accuracy"]>best[0]:
            best=(row["val_accuracy"],ep); torch.save({"model":model.state_dict(),"epoch":ep},out/f"{name}_best.pt")
    seconds = time.perf_counter() - t
    throughput = (len(loaders[0].dataset) * epochs) / max(seconds, 1e-9)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return hist, seconds, throughput, peak_gb

def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y,l in loader: ys.extend(y.numpy()); ps.extend(model(x.to(device),l.to(device)).sigmoid().cpu().numpy())
    return np.asarray(ys),np.asarray(ps)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="."); ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--max-len",type=int,default=256)
    ap.add_argument("--max-train",type=int,default=0); ap.add_argument("--smoke-test",action="store_true"); args=ap.parse_args(); seed_all()
    out=Path(args.output); (out/"outputs").mkdir(parents=True,exist_ok=True); (out/"checkpoints").mkdir(exist_ok=True); (out/"data_processed").mkdir(exist_ok=True)
    if args.smoke_test: args.epochs=1; args.max_train=5000; args.batch_size=64
    ds=load_dataset("fancyzhx/yelp_polarity"); split=ds["train"].train_test_split(test_size=.1,seed=SEED,stratify_by_column="label")
    rows={"train":split["train"],"validation":split["test"],"test":ds["test"]}
    if args.max_train: rows["train"]=rows["train"].select(range(min(args.max_train,len(rows["train"]))))
    counts=Counter(t for r in rows["train"] for t in tokenize(r["text"]))
    vocab={"<PAD>":0,"<UNK>":1}
    vocab.update({t:i for i,(t,c) in enumerate(sorted(((t,c) for t,c in counts.items() if c>=5),key=lambda z:(-z[1],z[0])),2)})
    json.dump(vocab,open(out/"data_processed/word_to_idx.json","w")); json.dump({"sid":SID,"seed":SEED,"max_len":args.max_len,"vocab_size":len(vocab)},open(out/"data_processed/config.json","w"),indent=2)
    arrays={k:encode(v,vocab,args.max_len) for k,v in rows.items()}
    test_texts=[str(row["text"]) for row in rows["test"]]
    pd.DataFrame([
        {"split":k,"examples":len(v),
         "missing_text":sum(row["text"] is None for row in v),
         "empty_text":sum(not str(row["text"]).strip() for row in v),
         "invalid_label":sum(int(row["label"]) not in (0,1) for row in v)}
        for k,v in rows.items()
    ]).to_csv(out/"outputs/data_quality.csv",index=False)
    pd.DataFrame([
        {"label":label,"count":int(sum(int(row["label"])==label for row in rows["train"]))}
        for label in (0,1)
    ]).to_csv(out/"outputs/class_distribution.csv",index=False)
    def loader(k,shuffle):
        x,y,l=arrays[k]; return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(l)),batch_size=args.batch_size,shuffle=shuffle,pin_memory=torch.cuda.is_available())
    tr,va,te=loader("train",True),loader("validation",False),loader("test",False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:",device,"GPU:",torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")
    specs={"baseline_cnn":CNNClassifier(len(vocab)),"experimental_bilstm_attention":BiLSTMAttention(len(vocab)),"experimental_bigru_pool":GRUMaxPool(len(vocab))}
    allrows=[]; preds={}
    for name,m in specs.items():
        h,secs,throughput,peak_gb=train_one(m.to(device),name,(tr,va),device,args.epochs,out/"checkpoints"); pd.DataFrame(h).to_csv(out/f"{name}_training_history.csv",index=False)
        best_state=torch.load(out/"checkpoints"/f"{name}_best.pt",map_location=device)
        m.load_state_dict(best_state["model"])
        y,p=predict(m,te,device); d,pred=metrics(y,p); d.update(model=name,training_seconds=secs,examples_per_sec=throughput,peak_memory_gb=peak_gb,parameters=sum(x.numel() for x in m.parameters())); allrows.append(d); preds[name]=(y,p,pred)
        pd.DataFrame(confusion_matrix(y,pred)).to_csv(out/"outputs"/f"{name}_confusion_matrix.csv",index=False)
    pd.DataFrame(allrows).to_csv(out/"outputs/core_test_metrics.csv",index=False)
    base=preds["baseline_cnn"][2]; mrows=[]
    for name in list(specs)[1:]:
        y=preds[name][0]; q=preds[name][2]; b=((base==y)&(q!=y)).sum(); c=((base!=y)&(q==y)).sum()
        mrows.append({"comparison":name,"b":int(b),"c":int(c),"p_value":float(binomtest(min(b,c),b+c).pvalue if b+c else 1.)})
    pd.DataFrame(mrows).to_csv(out/"outputs/mcnemar_tests.csv",index=False)
    ci=[]; slice_rows=[]; rng=np.random.default_rng(SEED)
    for name,(y,p,pred) in preds.items():
        vals=[]
        for _ in range(300):
            ix=rng.integers(0,len(y),len(y)); md,_=metrics(y[ix],p[ix]); vals.append([md["accuracy"],md["f1_macro"],md["mcc"]])
        a=np.asarray(vals); ci.append({"model":name,"accuracy_low":np.percentile(a[:,0],2.5),"accuracy_high":np.percentile(a[:,0],97.5),"macro_f1_low":np.percentile(a[:,1],2.5),"macro_f1_high":np.percentile(a[:,1],97.5),"mcc_low":np.percentile(a[:,2],2.5),"mcc_high":np.percentile(a[:,2],97.5)})
        lengths=arrays["test"][2]
        for s,mask in {"short_0_64":lengths<=64,"medium_65_128":(lengths>64)&(lengths<=128),"long_129_plus":lengths>128}.items():
            md,_=metrics(y[mask],p[mask]); slice_rows.append({"model":name,"slice":s,"n":int(mask.sum()),"macro_f1":md["f1_macro"],"error_rate":float((pred[mask]!=y[mask]).mean())})
    pd.DataFrame(ci).to_csv(out/"outputs/bootstrap_95ci.csv",index=False); pd.DataFrame(slice_rows).to_csv(out/"outputs/slice_robustness_metrics.csv",index=False)
    name="experimental_bilstm_attention"; y,p,pred=preds[name]; err=np.where(pred!=y)[0]; rows20=[]
    fp = err[(y[err] == 0) & (pred[err] == 1)]
    fn = err[(y[err] == 1) & (pred[err] == 0)]
    groups = [
        ("confident_false_positive", fp[np.argsort(-p[fp])[:5]]),
        ("confident_false_negative", fn[np.argsort(p[fn])[:5]]),
        ("near_threshold_error", err[np.argsort(abs(p[err] - .5))[:5]]),
    ]
    for cat,ix in groups:
        for i in ix: rows20.append({"index":int(i),"category":cat,"label":int(y[i]),"prediction":int(pred[i]),"prob_positive":float(p[i]),"processed_length":int(arrays["test"][2][i]),"text":test_texts[i],"error_type":"","observation":"","testable_fix":""})
    used={r["index"] for r in rows20}
    for i in err:
        if i not in used and len(rows20)<20: rows20.append({"index":int(i),"category":"slice_specific_failure","label":int(y[i]),"prediction":int(pred[i]),"prob_positive":float(p[i]),"processed_length":int(arrays["test"][2][i]),"text":test_texts[i],"error_type":"","observation":"","testable_fix":""})
    pd.DataFrame(rows20).to_csv(out/"outputs/manual_20_error_review.csv",index=False); print("Completed outputs in",out)
if __name__=="__main__": main()
