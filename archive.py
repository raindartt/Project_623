# used to concatenate all the text first then preprocess it
df_yn_tmp['Text'] = np.empty(510, dtype=str)
for col in YN_TXT_COLS:
    df_yn_tmp['Text'] += df_yn_tmp[col]
display(df_yn_tmp.info())
display(df_yn_tmp['Text'][0][:1000])

df_yn_tmp['Text'] = np.empty(510, dtype=str)
for col in YN_TXT_COLS:
    df_yn_tmp['Text'] += df_yn_tmp[col]
display(df_yn_tmp.info())
display(df_yn_tmp['Text'][0][:1000])



#### Trying word2vec
As demonstrated by https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/

Results were not useful

# from https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
# size instead of vector_size if using gensim 3.8.3 instead of 4
aux.word2vec_demo(names=df_master['Filename'], sentences=df_yn_txt, vector_size=100, workers=1, seed=SEED)


## CountVectorizer
## do not like because it has its own text preprocessing, so I'd have to pass in raw text
vectorizer = CountVectorizer(max_features=2500, min_df=5, max_df=0.7)
# uses the unproccessed data['sent'] instead of tokenized x_txt
x_count = vectorizer.fit_transform(data['sent']).toarray()

display(x_count[0].nonzero())
display(len(x_count[0].nonzero()[0]))




#### Trying TFidfTransformer
Restart with df_yn_txt and attempt TFidf as demonstrated by https://stackabuse.com/text-classification-with-python-and-scikit-learn/
    
# pick a class to act as the dependent variable y
yname = 'Cap On Liability-Answer'
y = df_01_ans[yname]
# display(y)

reload(aux)
rfc = aux.randomforest_demo(sentences=df_yn_txt, y=y, ynames=[yname], seed=SEED)

display(rfc.feature_importances_)