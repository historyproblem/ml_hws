"""Descriptive HW4 features: targets through 2024Q1, test features only.

No model fitting or held-out Q2/Q3 outcomes. Use from project root:
  .venv/bin/python scripts/analyze_hw4_feature_eda.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hw4_inspections.features import build_features

OUT = ROOT / "analysis/hw4-features-20261116/eda"
RAW = ROOT / "data/hw4/raw"
LIMIT = "2024-01-01"


def read_development():
    # Never interpret target values in the closed quarters.
    with (RAW / "train.csv").open() as handle:
        records = [record for record in csv.DictReader(handle)
                   if record["forecast_period"] <= LIMIT]
    frame = pd.DataFrame(records)
    frame["target"] = frame["target"].astype(int)
    return frame


def wilson(successes, n):
    p = successes / n
    z = 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center - half, center + half


def rate_table(data, feature, values):
    tmp = data[["site_id", "forecast_period", "target"]].copy()
    tmp["level"] = values.astype(str).to_numpy()
    results = []
    for level, part in tmp.groupby("level", sort=False):
        n, total = len(part), int(part.target.sum())
        low, high = wilson(total, n)
        results.append({"feature": feature, "level": level, "n": n,
                        "sites": part.site_id.nunique(), "positives": total,
                        "rate": total/n, "ci_low": low, "ci_high": high})
    return pd.DataFrame(results)


def cut(series, bins, labels):
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(object).fillna("missing")


def code_associations(data, x):
    # Common strata, defined without test labels. Exposure is code present/absent.
    # The adjustment controls burden, visit count, risk category and quarter.
    burden = pd.qcut(x.history_violation_mean, 8, duplicates="drop").astype(str)
    visits = cut(x.history_n_visits, [-np.inf,1,2,3,5,np.inf], ["1","2","3","4-5","6+"])
    strata = pd.MultiIndex.from_arrays([data.forecast_period, x.risk_level, burden, visits])
    stratum = pd.Series(pd.factorize(strata)[0], index=x.index)
    y = data.target.astype(float).reset_index(drop=True)
    y_centered = y - y.groupby(stratum).transform("mean")
    sites = data.site_id.reset_index(drop=True)
    rows = []
    for code in range(1,45):
        for suffix in ["ever", "last"]:
            name = f"code{code:02d}_{suffix}"
            flag = x[name].astype(float).reset_index(drop=True)
            resx = flag - flag.groupby(stratum).transform("mean")
            denom = float(np.square(resx).sum())
            n1, n0 = int(flag.sum()), int((flag == 0).sum())
            naive = float(y[flag==1].mean() - y[flag==0].mean()) if n1 and n0 else None
            beta = float((resx*y_centered).sum()/denom) if denom else None
            if denom:
                score = resx * (y_centered - beta*resx)
                sums = score.groupby(sites).sum()
                groups = len(sums)
                se = np.sqrt((groups/(groups-1)) * np.square(sums).sum())/denom
                low, high = beta - 1.96*se, beta + 1.96*se
            else:
                low = high = se = None
            quarter_diffs = []
            for quarter, index in data.groupby("forecast_period").groups.items():
                indices = np.asarray(list(index))
                f, yy = flag.iloc[indices], y.iloc[indices]
                quarter_diffs.append(float(yy[f==1].mean() - yy[f==0].mean())
                                     if f.sum() >= 10 and (f==0).sum() >= 10 else np.nan)
            positive = sum(v > 0 for v in quarter_diffs if np.isfinite(v))
            negative = sum(v < 0 for v in quarter_diffs if np.isfinite(v))
            rows.append({"feature": name, "code": code, "representation": suffix,
                         "n_present": n1, "n_absent": n0, "present_rate": float(y[flag==1].mean()) if n1 else None,
                         "absent_rate": float(y[flag==0].mean()) if n0 else None,
                         "unadjusted_risk_difference": naive, "adjusted_risk_difference": beta,
                         "cluster_se": se, "adjusted_ci_low": low, "adjusted_ci_high": high,
                         "overlap_rows": int((resx != 0).sum()), "positive_quarters": positive,
                         "negative_quarters": negative})
    return pd.DataFrame(rows)


def drift_table(reference, test, cats):
    rows=[]
    for col in reference:
        a,b=reference[col],test[col]
        if col in cats:
            aa=a.fillna("<missing>").astype(str)
            bb=b.fillna("<missing>").astype(str)
        else:
            values=a.dropna().astype(float)
            if values.nunique() <= 12:
                aa=a.fillna(-1e20).astype(str)
                bb=b.fillna(-1e20).astype(str)
            else:
                edges=np.unique(np.quantile(values, np.linspace(0,1,11)))
                edges[0],edges[-1]=-np.inf,np.inf
                aa=pd.cut(a,bins=edges,include_lowest=True).astype(str)
                bb=pd.cut(b,bins=edges,include_lowest=True).astype(str)
        levels=sorted(set(aa)|set(bb))
        ca=aa.value_counts().reindex(levels,fill_value=0).astype(float)+.5
        cb=bb.value_counts().reindex(levels,fill_value=0).astype(float)+.5
        pa,pb=ca/ca.sum(),cb/cb.sum()
        psi=float(((pb-pa)*np.log(pb/pa)).sum())
        midpoint=(pa+pb)/2
        js=float((pa*np.log(pa/midpoint)).sum()/2+(pb*np.log(pb/midpoint)).sum()/2)
        ks=float(ks_2samp(a.dropna(),b.dropna()).statistic) if col not in cats else None
        rows.append({"feature":col,"psi":psi,"jensen_shannon":js,"ks_statistic":ks,
                     "reference_missing":float(a.isna().mean()),"test_missing":float(b.isna().mean()),
                     "reference_unique":int(a.nunique()),"test_unique":int(b.nunique()),
                     "calendar_feature":col in ["forecast_year","forecast_quarter","cutoff_gap_days"]})
    return pd.DataFrame(rows).sort_values("psi",ascending=False)


def stratified_difference(data, exposure, strata):
    """Descriptive within-stratum difference, with site-cluster standard error."""
    sid=pd.Series(pd.factorize(pd.MultiIndex.from_arrays(strata))[0],index=data.index)
    yy=data.target.astype(float)
    xx=exposure.astype(float)
    rx=xx-xx.groupby(sid).transform("mean")
    ry=yy-yy.groupby(sid).transform("mean")
    denom=float(np.square(rx).sum())
    if not denom:return {"adjusted_difference":None,"ci_low":None,"ci_high":None,"overlap_rows":0}
    beta=float((rx*ry).sum()/denom)
    score=rx*(ry-beta*rx)
    sums=score.groupby(data.site_id).sum()
    se=float(np.sqrt(len(sums)/(len(sums)-1)*np.square(sums).sum())/denom)
    return {"adjusted_difference":beta,"ci_low":beta-1.96*se,"ci_high":beta+1.96*se,"overlap_rows":int((rx!=0).sum())}


def blank_and_outcome_analysis(data, dev, history):
    keys=data[["site_id","forecast_period","history_before"]].copy()
    keys["_row"]=np.arange(len(data))
    h=keys.merge(history,on=["site_id","forecast_period"],how="inner")
    h=h.loc[pd.to_datetime(h.inspection_week)<pd.to_datetime(h.history_before)].copy()
    h=h.sort_values(["_row","inspection_week"])
    h["blank_codes"]=h.violation_codes.isna()|h.violation_codes.fillna("").astype(str).str.strip().eq("")
    group=h.groupby("_row")
    last=group.tail(1).set_index("_row")
    routine=h.loc[h.inspection_type=="routine"].groupby("_row").tail(1).set_index("_row")
    flags={"last_codes_blank":last.blank_codes.reindex(data.index),
           "last_routine_codes_blank":routine.blank_codes.reindex(data.index).fillna("no routine"),
           "any_blank_code_visit":group.blank_codes.any().reindex(data.index),
           "all_code_visits_blank":group.blank_codes.all().reindex(data.index)}
    pd.concat([rate_table(data,name,flag) for name,flag in flags.items()],ignore_index=True).to_csv(OUT/"blank_code_risk_tables.csv",index=False)
    h.groupby(["blank_codes","result"]).agg(visits=("_row","size"),sites=("site_id","nunique")).reset_index().to_csv(OUT/"blank_codes_historical_outcome.csv",index=False)
    cross=[]
    for prefix,result_col,count_col in [("last","lag1_result","lag1_violation_count"),("last_routine","last_routine_result","last_routine_violation_count")]:
        bucket=cut(dev[count_col],[-np.inf,0,1,3,5,8,12,np.inf],["blank/0 recorded","1","2-3","4-5","6-8","9-12",">12"])
        temp=data[["site_id","forecast_period","target"]].copy()
        temp["last_outcome"]=dev[result_col]
        temp["burden_bucket"]=bucket
        g=temp.groupby(["burden_bucket","last_outcome"],observed=True).agg(n=("target","size"),positives=("target","sum"),sites=("site_id","nunique")).reset_index()
        g["future_fail_rate"]=g.positives/g.n;g["visit"]=prefix;cross.append(g)
    pd.concat(cross,ignore_index=True).to_csv(OUT/"outcome_vs_burden_strata.csv",index=False)
    assoc=[]
    for result_col,count_col in [("lag1_result","lag1_violation_count"),("last_routine_result","last_routine_violation_count")]:
        for contrast in ["fail","conditional"]:
            take=dev[result_col].isin(["pass",contrast])
            dd=data.loc[take].reset_index(drop=True)
            xx=dev.loc[take].reset_index(drop=True)
            flag=(xx[result_col]==contrast).astype(float)
            meanburden=pd.qcut(xx.history_violation_mean,8,duplicates="drop").astype(str)
            burden=cut(xx[count_col],[-np.inf,0,1,3,5,8,12,np.inf],["0","1","2-3","4-5","6-8","9-12",">12"])
            visits=cut(xx.history_n_visits,[-np.inf,1,2,3,5,np.inf],["1","2","3","4-5","6+"])
            for adjusted in [False,True]:
                strata=[dd.forecast_period,xx.risk_level]
                if adjusted:strata += [meanburden,burden,visits]
                assoc.append({"feature":result_col,"contrast":contrast+" minus pass","burden_adjusted":adjusted,
                              "n_contrast":int(flag.sum()),"n_pass":int((flag==0).sum()),
                              **stratified_difference(dd,flag,strata)})
    pd.DataFrame(assoc).to_csv(OUT/"outcome_adjusted_associations.csv",index=False)
    return {"blank_history_visit_fraction":float(h.blank_codes.mean()),"last_blank_fraction":float(last.blank_codes.mean()),
            "semantics":"Blank/NaN violation_codes means no recorded tokens in supplied CSV, not established absence of violations or established missing-at-random."}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=read_development().reset_index(drop=True)
    test=pd.read_csv(RAW/"test_student.csv")
    history=pd.read_csv(RAW/"inspection_history.csv")
    rows=pd.concat([data.drop(columns="target"),test],ignore_index=True)
    x,cats=build_features(rows,history)
    dev=x.iloc[:len(data)].copy().reset_index(drop=True)
    xt=x.iloc[len(data):].copy().reset_index(drop=True)
    assert data.forecast_period.max()==LIMIT
    bins={
        "history_violation_mean":cut(dev.history_violation_mean,[-np.inf,0,1,2,3,5,8,12,np.inf],["0","(0,1]","(1,2]","(2,3]","(3,5]","(5,8]","(8,12]",">12"]),
        "lag1_violation_count":cut(dev.lag1_violation_count,[-np.inf,0,1,3,5,8,12,np.inf],["0","1","2-3","4-5","6-8","9-12",">12"]),
        "history_n_visits":cut(dev.history_n_visits,[-np.inf,1,2,3,5,np.inf],["1","2","3","4-5","6+"]),
        "lag1_age_days":cut(dev.lag1_age_days,[-np.inf,30,90,180,365,np.inf],["<=30","31-90","91-180","181-365",">365"]),
        "last_routine_age_days":cut(dev.last_routine_age_days,[-np.inf,90,180,365,540,np.inf],["<=90","91-180","181-365","366-540",">540"]),
        "history_fail_rate":cut(dev.history_fail_rate,[-np.inf,0,.2,.4,.6,.8,np.inf],["0","(0,.2]","(.2,.4]","(.4,.6]","(.6,.8]",">.8"]),
        "had_reinspection":(dev.history_reinspection_count>0).astype(str),
        "had_complaint":(dev.type_complaint_count>0).astype(str),
        "lag1_result":dev.lag1_result,
        "last_routine_result":dev.last_routine_result,
        "lag1_inspection_type":dev.lag1_inspection_type,
        "risk_level":dev.risk_level,"facility_type":dev.facility_type,"area_id":dev.area_id,
    }
    tables=pd.concat([rate_table(data,name,values) for name,values in bins.items()],ignore_index=True)
    tables.to_csv(OUT/"risk_tables.csv",index=False)
    temporal=[]
    for name,values in bins.items():
        for q,indices in data.groupby("forecast_period").groups.items():
            r=rate_table(data.loc[indices],name,values.loc[indices])
            r["quarter"]=q
            temporal.append(r)
    pd.concat(temporal,ignore_index=True).to_csv(OUT/"risk_by_quarter.csv",index=False)
    profile=[]
    signatures={}
    for col in dev:
        signatures.setdefault(hashlib.sha256(pd.util.hash_pandas_object(dev[col],index=False).values.tobytes()).hexdigest(),[]).append(col)
        profile.append({"feature":col,"categorical":col in cats,"unique":int(dev[col].nunique()),
                        "missing_fraction":float(dev[col].isna().mean()),"test_missing_fraction":float(xt[col].isna().mean())})
    pd.DataFrame(profile).sort_values("missing_fraction",ascending=False).to_csv(OUT/"feature_profile.csv",index=False)
    duplicates=[columns for columns in signatures.values() if len(columns)>1]
    numeric=dev.select_dtypes(include="number")
    corr=numeric.corr(method="spearman")
    near=[]
    for i,col in enumerate(corr):
        for other in corr.columns[i+1:]:
            value=corr.loc[col,other]
            if abs(value)>.97:
                near.append({"left":col,"right":other,"spearman":value})
    pd.DataFrame(near,columns=["left","right","spearman"]).sort_values("spearman",key=lambda z:z.abs(),ascending=False).to_csv(OUT/"high_correlations.csv",index=False)
    associations=code_associations(data,dev)
    associations.to_csv(OUT/"code_associations.csv",index=False)
    blanks=blank_and_outcome_analysis(data,dev,history)
    latest=dev.loc[data.forecast_period==LIMIT]
    drift=drift_table(latest,xt,cats)
    drift.to_csv(OUT/"feature_drift_q1_to_test.csv",index=False)
    drift_table(dev,xt,cats).to_csv(OUT/"feature_drift_all_development_to_test.csv",index=False)
    summary={"scope":"descriptive, noncausal associations; no models fitted; code meanings unassigned",
             "target_quarters_max":LIMIT,"closed_2024q2_q3_targets_used":False,"test_labels_used":False,
             "development_rows":len(data),"test_rows":len(test),"positive_rate":float(data.target.mean()),
             "features":len(dev.columns),"constant_features":[c for c in dev if dev[c].nunique(dropna=False)<=1],
             "exact_duplicate_columns":duplicates,"near_correlations_gt_097":len(near),
             "uncertainty":"Risk-bin CIs are nominal Wilson, not cluster-corrected. Code adjusted associations use site-cluster SE, not causal effects; exploratory multiple comparisons.",
             "code_adjustment":"Within-stratum mean difference: quarter × risk × historical mean-code-count octile × visit-count bucket; weighting by n_present*n_absent/n. Cluster SE by site.",
             "blank_codes":blanks,
             "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/"train.csv",RAW/"test_student.csv",RAW/"inspection_history.csv",ROOT/"src/hw4_inspections/features.py",Path(__file__)]}}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,name,title in [(axes[0,0],"history_violation_mean","Среднее число записанных кодов за визит"),(axes[0,1],"last_routine_result","Последняя плановая проверка"),(axes[1,0],"lag1_age_days","Дней с последней проверки")]:
        tab=tables.loc[tables.feature==name].copy()
        levels=list(pd.unique(bins[name].astype(str)))
        if hasattr(bins[name],"cat"):
            levels=list(bins[name].cat.categories)
        if name=="history_violation_mean": levels=["0","(0,1]","(1,2]","(2,3]","(3,5]","(5,8]","(8,12]",">12"]
        if name=="lag1_age_days": levels=["<=30","31-90","91-180","181-365",">365"]
        tab=tab.set_index("level").reindex(levels).dropna(subset=["rate"])
        ax.errorbar(np.arange(len(tab)),100*tab.rate,yerr=100*np.array([tab.rate-tab.ci_low,tab.ci_high-tab.rate]),fmt="o",capsize=3,color="#2463a3")
        ax.set_xticks(np.arange(len(tab)),tab.index,rotation=25,ha="right")
        ax.set_title(title); ax.set_ylabel("Доля будущих провалов, %")
        ax.axhline(100*data.target.mean(),color="gray",ls="--",lw=1)
        for index,(_,r) in enumerate(tab.iterrows()):ax.annotate(f"n={int(r.n)}",(index,100*r.rate),xytext=(0,10),textcoords="offset points",ha="center",fontsize=8)
    show=associations.query("representation=='ever' and n_present>=100 and overlap_rows>=100").copy()
    show=show.iloc[np.argsort(np.abs(show.adjusted_risk_difference))[-8:]].sort_values("adjusted_risk_difference")
    ax=axes[1,1]
    ax.errorbar(100*show.adjusted_risk_difference,np.arange(len(show)),xerr=100*np.array([show.adjusted_risk_difference-show.adjusted_ci_low,show.adjusted_ci_high-show.adjusted_risk_difference]),fmt="o",capsize=3,color="#0f8b73")
    ax.set_yticks(np.arange(len(show)),show.feature)
    ax.axvline(0,color="gray",ls="--",lw=1);ax.set_xlabel("Разница риска после стратификации, п.п.")
    ax.set_title("Коды: поправка на нагрузку, риск и квартал")
    fig.suptitle("HW4: наблюдаемые связи признаков с исходом\nТолько train 2022Q1–2024Q1; будущий исход = первая routine-проверка",fontsize=14)
    fig.text(.01,.01,"Описательные связи, не причинные эффекты. Интервалы первых 3 графиков: nominal Wilson; кодов: cluster SE по площадке.\nКоды выбраны после просмотра связей; множественные сравнения и корреляция признаков требуют проверки абляциями.",fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.93]);fig.savefig(OUT/"feature_relationships.png",dpi=160);fig.savefig(OUT/"feature_relationships.svg");plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("\nRISK TABLES\n",tables.to_string(index=False))
    print("\nADJUSTED CODE ASSOCIATIONS\n",associations.query("representation=='ever'").sort_values("adjusted_risk_difference",ascending=False).head(12).to_string(index=False))
    print("\nDRIFT\n",drift.head(15).to_string(index=False))


if __name__=="__main__":
    main()
