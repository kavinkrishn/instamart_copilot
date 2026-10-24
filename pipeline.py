
from pathlib import Path
import os, re, json, time, random, sqlite3, argparse, statistics
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import mock_llm

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; RAW=DATA/'raw'; CLEAN=DATA/'clean'; OUT=ROOT/'outputs'; STATE=ROOT/'state'
for p in [CLEAN,OUT,STATE,ROOT/'sql']: p.mkdir(parents=True,exist_ok=True)
SNAP=pd.Timestamp('2025-10-01 00:00:00')
CUTOFF=pd.Timestamp('2025-09-01')

def part_a():
    c=pd.read_csv(RAW/'customers.csv', keep_default_na=False)
    p=pd.read_csv(RAW/'products.csv')
    o=pd.read_csv(RAW/'orders.csv', keep_default_na=False)
    # clean customers
    c['city']=c.city.astype(str).str.strip().str.title()
    c['gender']=c.gender.astype(str).str.strip().str.lower().map({'male':'male','m':'male','female':'female','f':'female'}).fillna('unknown')
    c['age']=pd.to_numeric(c.age,errors='coerce')
    c.loc[~c.age.between(10,100),'age']=np.nan
    c['signup_date']=pd.to_datetime(c.signup_date)
    c['age']=c['age'].fillna(c.groupby('city')['age'].transform('median')).round().astype(int)
    # clean orders
    o['order_ts']=pd.to_datetime(o.order_ts)
    o=o.drop_duplicates()
    o=o.sort_values('order_ts').drop_duplicates('order_id',keep='last').copy()
    o['quantity']=pd.to_numeric(o.quantity,errors='coerce')
    o['discount_pct']=pd.to_numeric(o.discount_pct,errors='coerce')
    o['order_amount']=pd.to_numeric(o.order_amount,errors='coerce')
    o=o.merge(p[['product_id','price_inr','category']],on='product_id',how='left')
    miss=o.order_amount.isna()
    o.loc[miss,'order_amount']=(o.loc[miss,'price_inr']*o.loc[miss,'quantity']*(1-o.loc[miss,'discount_pct']/100)).round(2)
    o['delivery_minutes']=pd.to_numeric(o.delivery_minutes,errors='coerce')
    o.loc[o.delivery_minutes<0,'delivery_minutes']=np.nan
    completed=o.status.eq('completed')
    med=o.loc[completed].groupby('category').delivery_minutes.transform('median')
    # category median lookup including all completed orders, including missing duration ignored
    medians=o.loc[completed].groupby('category').delivery_minutes.median()
    fillmask=completed & o.delivery_minutes.isna()
    o.loc[fillmask,'delivery_minutes']=o.loc[fillmask,'category'].map(medians)
    o.loc[o.status.eq('cancelled'),'delivery_minutes']=np.nan
    o=o.drop(columns=['price_inr','category'])
    c['signup_date']=c.signup_date.dt.strftime('%Y-%m-%d')
    o['order_ts']=o.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    c.to_csv(CLEAN/'customers.csv',index=False)
    p.to_csv(CLEAN/'products.csv',index=False)
    o.to_csv(CLEAN/'orders.csv',index=False)
    db=sqlite3.connect(DATA/'instamart.db')
    c.to_sql('customers',db,if_exists='replace',index=False)
    p.to_sql('products',db,if_exists='replace',index=False)
    o.to_sql('orders',db,if_exists='replace',index=False)
    db.close()
    # SQL queries
    queries = [
"""SELECT customer_id, SUM(order_amount) total FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;""",
"""SELECT c.city,p.category,SUM(o.order_amount) revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;""",
"""SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);""",
"""SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;""",
"""WITH spend AS (SELECT customer_id,SUM(order_amount) total FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total) av FROM spend) SELECT spend.* FROM spend,avg_spend WHERE spend.total>avg_spend.av;""",
"""WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM customers c JOIN orders o USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT customer_id,city,total_spend,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rnk,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rnk,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) rn FROM spend;""",
"""WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) cumulative_revenue FROM daily ORDER BY day;"""
    ]
    (ROOT/'sql'/'queries.sql').write_text('\n\n'.join(q for q in queries),encoding='utf-8')
    checks = [
        len(c)==1200, len(p)==40, len(o)==12309, o.order_id.nunique()==12309,
        (o.status=='completed').sum()==11110, (o.status=='cancelled').sum()==1199,
        not o.order_amount.isna().any(), o.loc[o.status.eq('completed'),'delivery_minutes'].notna().all(),
        o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().all(),
        round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2)==2998668.80,
        c.age.notna().all(), c.city.nunique()==6
    ]
    if not all(checks): print('CHECK WARNINGS',[(i,x) for i,x in enumerate(checks) if not x])
    else: print('ALL CHECKPOINTS PASSED')
    db=sqlite3.connect(DATA/'instamart.db')
    with open(ROOT/'sql'/'queries.sql') as f: qs=f.read().split(';')
    for i,q in enumerate(qs):
        if q.strip():
            df=pd.read_sql_query(q,db); print(f'Q{i+1}: rows={len(df)}')
    db.close()
    # exploratory summary for README
    done=o[o.status.eq('completed')].copy()
    done['order_amount']=pd.to_numeric(done.order_amount)
    print('Part A summary:',json.dumps({'completed_revenue':round(done.order_amount.sum(),2),'mean':round(done.order_amount.mean(),2),'median':round(done.order_amount.median(),2),'skew':round(done.order_amount.skew(),2),'q1':done.order_amount.quantile(.25),'q3':done.order_amount.quantile(.75)},default=float))

def part_b():
    c=pd.read_csv(CLEAN/'customers.csv')
    o=pd.read_csv(CLEAN/'orders.csv',parse_dates=['order_ts'])
    done=o[o.status.eq('completed')].copy()
    rfm=done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(SNAP-rfm.last_order).dt.days
    rfm=rfm[['customer_id','recency','frequency','monetary']]
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']])
    scores=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
        scores.append({'k':k,'inertia':round(km.inertia_,1),'silhouette':round(silhouette_score(X,km.labels_),3)})
    print('KMeans:',scores)
    k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X); rfm['segment_id']=km.labels_
    prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).sort_values('monetary',ascending=False)
    names={}
    for ix,row in prof.iterrows():
        if row.monetary==prof.monetary.max(): name='Premium'
        elif row.frequency==prof.frequency.max(): name='Frequent Essentials'
        elif row.recency==prof.recency.min(): name='Recently Active'
        else: name='At-Risk / Occasional'
        names[ix]=name
    rfm['segment_name']=rfm.segment_id.map(names)
    rfm['is_premium']=rfm.segment_name.eq('Premium')
    rfm.to_csv(OUT/'customer_segments.csv',index=False)
    print('Segments:',prof.to_dict('index'), 'PCA:',PCA(n_components=2).fit(X).explained_variance_ratio_.tolist())
    # leakage-safe lapse cohort
    oo=o[o.order_ts<CUTOFF].copy()
    all_pre=oo[oo.status.isin(['completed','cancelled'])]
    comp=oo[oo.status.eq('completed')].copy()
    last=comp.groupby('customer_id').order_ts.max()
    cohort=last.index
    features=pd.DataFrame({'customer_id':cohort})
    features['recency']=(CUTOFF-pd.to_datetime(last.reindex(cohort).to_numpy())).days if False else (CUTOFF-last.reindex(cohort).reset_index(drop=True)).dt.days
    agg=comp.groupby('customer_id').agg(frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'))
    features=features.merge(agg,on='customer_id',how='left')
    canc=all_pre.assign(is_cancelled=all_pre.status.eq('cancelled')).groupby('customer_id').is_cancelled.agg(['sum','count'])
    features['cancellation_rate']=features.customer_id.map((canc['sum']/canc['count']).to_dict()).fillna(0)
    fav=comp.merge(pd.read_csv(CLEAN/'products.csv')[['product_id','category']],on='product_id').groupby('customer_id').category.agg(lambda s:s.value_counts().index[0])
    features['favourite_category']=features.customer_id.map(fav)
    features=features.merge(c[['customer_id','city','age','gender','signup_date']],on='customer_id',how='left')
    features['tenure_days']=(CUTOFF-pd.to_datetime(features.signup_date)).dt.days
    sep=done[(done.order_ts>=CUTOFF)&(done.order_ts<pd.Timestamp('2025-10-01'))].groupby('customer_id').size()
    features['lapsed']=(~features.customer_id.isin(sep.index)).astype(int)
    features=features.replace([np.inf,-np.inf],np.nan)
    for col in ['avg_discount','avg_delivery','favourite_category']:
        features[col]=features[col].fillna(features[col].mode().iloc[0] if features[col].notna().any() else 0)
    for col in ['frequency','monetary','recency','cancellation_rate','tenure_days','age']:
        features[col]=features[col].fillna(features[col].median())
    features['city']=features.city.fillna(features.city.mode()[0]); features['gender']=features.gender.fillna('unknown')
    Xdf=features.drop(columns=['customer_id','lapsed','signup_date'])
    y=features.lapsed
    Xtr,Xte,ytr,yte,idxtr,idxte=train_test_split(Xdf,y,features.customer_id,test_size=.2,stratify=y,random_state=42)
    num=Xdf.select_dtypes(include=np.number).columns.tolist()
    cat=['city','gender','favourite_category']
    try: enc=OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)
    except TypeError: enc=OneHotEncoder(drop='first',handle_unknown='ignore',sparse=False)
    prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',enc)]),cat)])
    logpipe=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000))])
    loggrid=GridSearchCV(logpipe,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
    rfpipe=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    rfgrid=GridSearchCV(rfpipe,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
    def metrics(model,label):
        pred=model.predict(Xte)
        print(label, 'best',model.best_params_,'cvf1',round(model.best_score_,3),'cm',confusion_matrix(yte,pred).tolist(),'precision',precision_score(yte,pred,zero_division=0),'recall',recall_score(yte,pred,zero_division=0),'f1',f1_score(yte,pred,zero_division=0))
    metrics(loggrid,'Logistic'); metrics(rfgrid,'RandomForest')
    chosen=rfgrid.best_estimator_; pred=chosen.predict(Xte)
    joblib.dump(chosen,OUT/'lapse_model.joblib')
    assert np.array_equal(pred,joblib.load(OUT/'lapse_model.joblib').predict(Xte))
    # oversample training only for best logistic configuration
    bestlog=loggrid.best_estimator_
    train_encoded=bestlog.named_steps['prep'].transform(Xtr)
    counts=ytr.value_counts(); rng=np.random.RandomState(42)
    majority=int(counts.max()); parts=[np.arange(len(ytr))]
    pos=np.flatnonzero(ytr.to_numpy()==1)
    if len(pos):
        extra=rng.choice(pos,size=majority-int(counts.get(1,0)),replace=True)
        Xbal=np.vstack([train_encoded,train_encoded[extra]])
        ybal=np.concatenate([ytr.to_numpy(),ytr.to_numpy()[extra]])
        model=LogisticRegression(max_iter=1000, C=bestlog.named_steps['model'].C).fit(Xbal,ybal)
        yp=model.predict(bestlog.named_steps['prep'].transform(Xte))
        print('Balanced logistic',precision_score(yte,yp,zero_division=0),recall_score(yte,yp,zero_division=0),f1_score(yte,yp,zero_division=0))
    # top Premium in test set; illustrative post-hoc segment flag
    seg=rfm[['customer_id','is_premium']]
    probs=chosen.predict_proba(Xte)[:,list(chosen.classes_).index(1)]
    ret=pd.DataFrame({'customer_id':idxte.values,'lapse_probability':probs,'actually_lapsed':yte.values}).merge(seg,on='customer_id')
    ret=ret[ret.is_premium].sort_values('lapse_probability',ascending=False).head(10).drop(columns='is_premium')
    ret.to_csv(OUT/'retention_list.csv',index=False)
    print('Cohort',len(features),'lapsed',int(y.sum()),'train/test',len(Xtr),len(Xte),'premium retention true lapsed',int(ret.actually_lapsed.sum()))

EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
PHONE=re.compile(r'\b[6-9]\d{9}\b')
ORDER=re.compile(r'ORD-\d{6}')
ALLOWED={'billing','delivery','technical','fraud'}
def redact(s):
    counts={}
    for name,pat,repl in [('email',EMAIL,'[EMAIL]'),('phone',PHONE,'[PHONE]'),('order',ORDER,'[ORDER_ID]')]:
        s,n=pat.subn(repl,s); counts[name]=n
    return s,counts

def triage_run(prompt, budget=100000, outpath=None):
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    seg=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id').to_dict('index')
    mock_llm.reset(); rng=random.Random(42); used=0; rows=[]; totals={'email':0,'phone':0,'order':0}; retries=0
    for _,t in tickets.iterrows():
        text0=t['text']; status=''; cat=''; tokens=0
        if not text0.strip(): status='skipped_empty'
        elif mock_llm.est_tokens(text0)>400: status='flagged_long'
        else:
            safe,counts=redact(text0)
            for k,v in counts.items(): totals[k]+=v
            assert not EMAIL.search(safe) and not PHONE.search(safe) and not ORDER.search(safe)
            if used>=budget: status='budget_exhausted'
            else:
                messages=[{'role':'system','content':prompt},{'role':'user','content':safe}]
                attempt=0
                while True:
                    try:
                        resp=mock_llm.chat(messages); break
                    except mock_llm.TransientLLMError:
                        if attempt>=3: raise
                        wait=.05*(2**attempt)*rng.uniform(.9,1.15)
                        print(f"retry ticket={t.ticket_id} retry={attempt} wait={wait:.4f}s")
                        time.sleep(wait); attempt+=1; retries+=1
                tokens=resp.usage.total_tokens; used+=tokens
                raw=resp.choices[0].message.content or ''
                try:
                    data=json.loads(raw); cat0=data.get('category')
                    if cat0 in ALLOWED: cat=cat0; status='classified_json'
                    else: status='flagged_invalid_output'
                except Exception:
                    words=re.findall(r'[a-z]+',raw.lower())
                    found=[x for x in ALLOWED if x in words]
                    if len(found)==1: cat=found[0]; status='classified_keyword'
                    else: status='flagged_invalid_output'
        si=seg.get(t.customer_id,{})
        segment=si.get('segment_name','')
        premium=bool(si.get('is_premium',False))
        if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: route,priority='human_review','NORMAL'
        elif cat=='fraud': route,priority='security_team','HIGH'
        elif premium: route,priority='premium_desk','HIGH'
        else: route,priority='support_queue','NORMAL'
        rows.append({'ticket_id':t.ticket_id,'customer_id':t.customer_id,'status':status,'category':cat,'segment_name':segment,'priority':priority,'route':route,'tokens_used':tokens})
    df=pd.DataFrame(rows)
    if outpath: df.to_csv(outpath,index=False)
    truth=tickets[['ticket_id','true_category']]
    scored=df.merge(truth,on='ticket_id')
    valid=scored[scored.status.isin(['classified_json','classified_keyword'])]
    acc=float((valid.category==valid.true_category).mean()) if len(valid) else 0
    print('Triage',outpath,'status',df.status.value_counts().to_dict(),'classified',len(valid),'accuracy',round(acc,4),'tokens',used,'tokens_per_classified',round(used/max(1,len(valid)),2),'retries',retries,'redactions',totals)
    if 'zero_shot' in str(outpath):
        print('zero-shot confusion table')
        print(pd.crosstab(valid.true_category,valid.category).to_string())
    return df

def part_c():
    z='CLASSIFY. Classify the support ticket into exactly one category: billing, delivery, technical, fraud. Return JSON only in the form {"category":"...","reason":"..."}; do not add prose.'
    fs='CLASSIFY. Classify into billing, delivery, technical, fraud. Return JSON only in the form {"category":"...","reason":"..."}.\nEXAMPLES:\nTicket: unauthorized transaction on my card. Reply: {"category":"fraud","reason":"unauthorized transaction"}\nTicket: app crashes on login. Reply: {"category":"technical","reason":"app crash"}'
    triage_run(z,100000,OUT/'triage_zero_shot.csv')
    triage_run(fs,100000,OUT/'triage_few_shot.csv')
    triage_run(z,6000,OUT/'triage_budget_demo.csv')
    triage_run(fs,100000,OUT/'triage_results.csv')

# policy retrieval
def make_chunks(overlap=10):
    chunks=[]
    for path in sorted((DATA/'policies').glob('*.txt')):
        words=path.read_text(encoding='utf-8').split()
        step=40-overlap
        start=0
        while start<len(words):
            chunk=words[start:start+40]
            chunks.append({'source':path.name,'text':' '.join(chunk)})
            if start+40>=len(words): break
            start+=step
    return chunks
def retrieval_factory(overlap=10):
    chunks=make_chunks(overlap)
    vec=TfidfVectorizer()
    mat=vec.fit_transform([x['text'] for x in chunks])
    def retrieve(query,k=3):
        q=vec.transform([query]); sims=cosine_similarity(q,mat).ravel()
        ix=np.argsort(-sims,kind='stable')[:k]
        return [{'source':chunks[i]['source'],'score':float(sims[i]),'text':chunks[i]['text']} for i in ix]
    return chunks,retrieve
def policy_answer(question,retrieve):
    got=retrieve(question,3)
    context='\n'.join(f"[{x['source']}] {x['text']}" for x in got)
    msgs=[{'role':'system','content':'POLICY ASSISTANT: Answer only from supplied context. If the answer is not present, say I could not find that information in the policy documents.'},
          {'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
    res=mock_llm.chat(msgs)
    return res.choices[0].message.content,[x['source'] for x in got]

def part_d():
    chunks,retrieve=retrieval_factory(10)
    print('chunks',len(chunks),pd.Series([x['source'] for x in chunks]).value_counts().to_dict())
    ev=pd.read_csv(RAW/'policy_eval.csv',keep_default_na=False)
    good_doc=good_fact=fallback=0
    for _,r in ev.iterrows():
        ans,sources=policy_answer(r.question,retrieve)
        if r.expected_doc=='NONE':
            fallback+=ans.startswith('I could not find')
        else:
            good_doc+=r.expected_doc in sources
            good_fact+=r.expected_fact in ans
    print('RAG eval overlap10',good_doc,good_fact,fallback)
    _,ret0=retrieval_factory(0)
    gd=gf=fb=0
    for _,r in ev.iterrows():
        ans,sources=policy_answer(r.question,ret0)
        if r.expected_doc=='NONE': fb+=ans.startswith('I could not find')
        else: gd+=r.expected_doc in sources; gf+=r.expected_fact in ans
    print('RAG eval overlap0',gd,gf,fb)
    db=sqlite3.connect(DATA/'instamart.db')
    ledger=OUT/'refund_ledger.jsonl'; ledger.write_text('',encoding='utf-8')
    def get_order(order_id):
        row=db.execute("SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?",(order_id,)).fetchone()
        return {'error':'order not found'} if row is None else dict(zip(['order_id','status','order_amount','delivery_minutes'],row))
    def search_policy(query): return {'results':retrieve(query,3)}
    def calculate_refund(order_amount,delivery_minutes):
        return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
    def ledger_tool(tool_name,order_id,refund_inr):
        rec={'timestamp':datetime.now().isoformat(),'tool':tool_name,'order_id':order_id,'refund_inr':float(refund_inr)}
        with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
        return {'refund_inr':float(refund_inr)}
    def issue_refund(order_id,refund_inr): return ledger_tool('issue_refund',order_id,refund_inr)
    def escalate_to_human(order_id,refund_inr): return ledger_tool('escalate_to_human',order_id,refund_inr)
    toolmap={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
    tools_spec=[{'name':'get_order','description':'Look up an order by id.','parameters':{'order_id':'string'}},{'name':'search_policy','description':'Retrieve relevant policy chunks.','parameters':{'query':'string'}},{'name':'calculate_refund','description':'Calculate eligible refund.','parameters':{'order_amount':'number','delivery_minutes':'number'}},{'name':'issue_refund','description':'Record an automatic refund.','parameters':{'order_id':'string','refund_inr':'number'}},{'name':'escalate_to_human','description':'Send a large refund to human review.','parameters':{'order_id':'string','refund_inr':'number'}}]
    def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None,resume=False,initial=None):
        started=time.monotonic()
        if initial:
            messages=initial['messages']; tokens=initial['tokens_used']; steps=initial['steps']; tools_done=initial['tools_executed']; llm_calls=initial['llm_calls']
        else:
            messages=[{'role':'system','content':'AGENT. Use tools to resolve refund requests. Follow policy.'},{'role':'user','content':user_message}]
            tokens=0; steps=[]; tools_done=[]; llm_calls=0
        for iteration in range(max_iterations):
            if time.monotonic()-started>=time_limit_seconds:
                return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human agent with the steps completed so far.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens,'tools_executed':tools_done}
            resp=mock_llm.chat(messages,tools=tools_spec); llm_calls+=1; tokens+=resp.usage.total_tokens
            if tokens>token_limit:
                return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human agent; token limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens,'tools_executed':tools_done}
            msg=resp.choices[0].message
            if not msg.tool_calls:
                answer=msg.content or ''
                return {'status':answer.split(':',1)[0],'final_answer':answer,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens,'tools_executed':tools_done}
            tc=msg.tool_calls[0]; args=tc.arguments
            messages.append({'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':tc.name,'arguments':args}]})
            obs=toolmap[tc.name](**args)
            step={'tool':tc.name,'arguments':args,'observation':obs}
            steps.append(step); tools_done.append(tc.name)
            messages.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(obs)})
            rid=run_id or scenario_id
            checkpoint={'run_id':rid,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens,'step_number':len(steps),'steps':steps,'tools_executed':tools_done,'llm_calls':llm_calls,'user_message':user_message}
            (STATE/f'{rid}.json').write_text(json.dumps(checkpoint,default=str),encoding='utf-8')
            if crash_after and len(steps)==crash_after: raise RuntimeError('SIMULATED_CRASH')
        return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human agent with the steps completed so far.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens,'tools_executed':tools_done}
    def resume_agent(run_id):
        checkpoint=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
        return run_agent(checkpoint['user_message'],checkpoint['scenario_id'],run_id=run_id,initial=checkpoint)
    scenarios=pd.read_csv(RAW/'agent_scenarios.csv',keep_default_na=False)
    runs=[]
    for _,s in scenarios.iterrows():
        result=run_agent(s.message,s.scenario_id)
        record={k:result[k] for k in ['status','final_answer','llm_calls','tokens_used','steps']}
        record['scenario_id']=s.scenario_id
        runs.append(record)
    (OUT/'agent_runs.jsonl').write_text(''.join(json.dumps(x,default=str)+'\n' for x in runs),encoding='utf-8')
    print('Agent scenarios',[(x['scenario_id'],x['status'],x['llm_calls'],x['tokens_used']) for x in runs])
    # safety demonstrations
    s1=scenarios[scenarios.scenario_id.eq('S1')].iloc[0]
    print('token demo',run_agent(s1.message,'S1_TOKEN',token_limit=200)['status'])
    print('timeout demo',run_agent(s1.message,'S1_TIMEOUT',time_limit_seconds=0)['status'])
    # crash/resume demonstration with isolated run id; remove its prior checkpoint
    rid='S1_CRASH_RESUME'
    try:
        run_agent(s1.message,'S1',run_id=rid,crash_after=2)
    except RuntimeError: pass
    resumed=resume_agent(rid)
    print('crash-resume',resumed['status'],resumed['tools_executed'])
    db.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('part',choices=['A','B','C','D','all']); args=ap.parse_args()
    if not (RAW/'customers.csv').exists(): os.system(f'"{os.sys.executable}" "{ROOT/"generate_data.py"}')
    if args.part in ('A','all'): part_a()
    if args.part in ('B','all'): part_b()
    if args.part in ('C','all'): part_c()
    if args.part in ('D','all'): part_d()
if __name__=='__main__': main()
