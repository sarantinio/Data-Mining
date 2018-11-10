import pyltr.pyltr as pyltr
import pandas as pd
import read_data as rd
import pickle as pk
import numpy as np


def print_df(df):
    with open('ranking.txt', 'a') as the_file:
        for index, row in df.iterrows():
            the_file.write(str(int(row['srch_id'])) + str(',') + str(int(row['prop_id'])) + str('\n'))


df_train = pd.read_csv('trained_all_relevance.csv', nrows=1000000, header=0)
df_eval = pd.read_csv('trained_all_relevance.csv', skiprows=range(1, 1000000), nrows=200000, header=0)
df_test = pd.read_csv('trained_all_relevance.csv', skiprows=range(1, 1200000), header=0)

print(len(df_train.index))
print(len(df_test.index))
print(len(df_eval.index))

# train_data = rd.aggregate_comp(df_train)
# train_data = rd.filter_dates(train_data)
#
# eval_data = rd.aggregate_comp(df_eval)
# eval_data = rd.filter_dates(eval_data)
#
# test_data = rd.filter_dates(rd.aggregate_comp(df_test))
train_data = df_train
eval_data = df_eval
test_data = df_test

train_data = train_data.drop('visitor_hist_starrating', 1)
train_data = train_data.drop('visitor_hist_adr_usd', 1)
train_data = train_data.drop('srch_query_affinity_score', 1)
train_data = train_data.drop('gross_bookings_usd', 1)
train_data = train_data.drop('Unnamed: 0', 1)
train_data = train_data.drop('Unnamed: 0.1', 1)
train_data = train_data.drop('Unnamed: 0.1.1', 1)
train_data = train_data.drop('Unnamed: 0.1.1.1', 1)
# train_data = train_data.drop('Unnamed: 0.1.1.1.1', 1)

eval_data = eval_data.drop('visitor_hist_starrating', 1)
eval_data = eval_data.drop('visitor_hist_adr_usd', 1)
eval_data = eval_data.drop('srch_query_affinity_score', 1)
eval_data = eval_data.drop('gross_bookings_usd', 1)
eval_data = eval_data.drop('Unnamed: 0', 1)
eval_data = eval_data.drop('Unnamed: 0.1', 1)
eval_data = eval_data.drop('Unnamed: 0.1.1', 1)
eval_data = eval_data.drop('Unnamed: 0.1.1.1', 1)
# eval_data = eval_data.drop('Unnamed: 0.1.1.1.1', 1)

test_data = test_data.drop('visitor_hist_starrating', 1)
test_data = test_data.drop('visitor_hist_adr_usd', 1)
test_data = test_data.drop('srch_query_affinity_score', 1)
test_data = test_data.drop('gross_bookings_usd', 1)
test_data = test_data.drop('Unnamed: 0', 1)
test_data = test_data.drop('Unnamed: 0.1', 1)
test_data = test_data.drop('Unnamed: 0.1.1', 1)
test_data = test_data.drop('Unnamed: 0.1.1.1', 1)
# test_data = test_data.drop('Unnamed: 0.1.1.1.1', 1)

label_train = train_data['relevance_score']
train_data = train_data.drop('relevance_score', axis=1)
train_data = train_data.drop('booking_bool', axis=1)
train_data = train_data.drop('click_bool', axis=1)
train_group = train_data['srch_id']
# train_data.pop('srch_id')

label_eval = eval_data['relevance_score']
eval_data = eval_data.drop('relevance_score', axis=1)
eval_data = eval_data.drop('position', axis=1)
eval_data = eval_data.drop('booking_bool', axis=1)
eval_data = eval_data.drop('click_bool', axis=1)
eval_group = eval_data['srch_id']
# eval_data.pop('srch_id')


label_test = test_data['relevance_score']
test_data = test_data.drop('relevance_score', axis=1)
test_data = test_data.drop('position', axis=1)
test_data = test_data.drop('booking_bool', axis=1)
test_data = test_data.drop('click_bool', axis=1)
test_group = test_data['srch_id']
# test_data.pop('srch_id')

# test_group = test_data['srch_id']
# test_data.pop('srch_id')


metric = pyltr.metrics.NDCG(k=10)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    eval_data, label_eval, eval_group, metric=metric)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=500,
    learning_rate=0.0005,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=20,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(train_data, label_train, train_group, monitor=monitor)

Epred = model.predict(test_data)

for x in np.unique(test_group):
    #indices = test_data['srch_id'] == x
    this_id = test_data.loc[test_data['srch_id'] == x]
    this_scores = Epred[test_data['srch_id'] == x]
    this_id['scores'] = this_scores
    this_id.sort_values('scores')
    print_df(this_id)
    #print(x)

with open('Epred2.pkl', 'wb') as f:
    pk.dump(Epred, f)

test_group = np.array(test_group)

print('Random ranking:')
print('Random ranking:', metric.calc_mean_random(test_group, np.array(label_test)))
print('Our model:', metric.calc_mean(test_group, np.array(label_test), Epred))