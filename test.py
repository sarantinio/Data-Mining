import pandas as pd
from imputer import Imputer
impute = Imputer()


'''
columns = ["srch_room_count", "visitor_location_country_id", "srch_booking_window", \
           "click_bool", "prop_country_id", "date_time", "booking_bool", "prop_starrating",\
           "srch_length_of_stay", "prop_location_score1", "position", "random_bool",\
            "srch_children_count", "srch_saturday_night_bool", "srch_adults_count",\
            "prop_log_historical_price", "price_usd", "srch_id", "prop_brand_bool",\
            "promotion_flag", "srch_destination_id", "site_id", "prop_id"]


df = pd.read_csv('../Data/aggregate_comp_data_train.csv', header=0)
correlations = df.corr() #all correlations
#correlations.to_csv("../Data/aggregate-comp-correlations.csv")
strong_corr = (((correlations.abs()).unstack()).sort_values(kind="quicksort")).to_frame()
#strong_corr.to_csv("../Data/aggregate-comp-correlations-sorted.csv")

df = pd.read_csv('../Data/filtered-aggregated-test.csv', header=0, nrows=500000)
df2 = pd.read_csv('../Data/filtered-aggregated-test.csv', header=0, nrows=500000)

#x_filled_mice = fi.MICE().complete(df)

cols_to_impute = ["orig_destination_distance","prop_location_score2","prop_review_score","comp_inv","comp_rate","comp_rate_percent_diff"]
print("starting to impute")
for column_name in cols_to_impute:
    print("imputing column: "+ column_name)
    df2.update(pd.DataFrame(impute.knn(df, column=column_name, k=5), columns=df.keys()))
print("Saving to file...")
df2.to_csv("../Data/test-filtered-aggregated-imputed.csv")
print("Saved...")
#x_filled_fancy_knn = pd.DataFrame(fi.KNN(k=5).complete(df), columns=df.keys())
'''
#columns = ["comp_inv", "comp_rate", "comp_rate_percent_diff"]
#df = pd.read_csv('../Data/aggregate_comp_data_train.csv', header=0)
#correlations = pd.read_csv("../Data/aggregate-comp-correlations.csv" ,header=0, usecols=columns)
#sorted_correlations = pd.read_csv("../Data/aggregate-comp-correlations-sorted.csv", header = 0, usecols = columns)

# df = pd.read_csv('../Data/train-filtered-aggregated.csv', header=0, nrows=500000)
# df_2 = pd.read_csv('../Data/train-filtered-aggregated.csv', header=0, nrows=500000)
#
# cols_to_impute = ["orig_destination_distance","prop_location_score2","prop_review_score"]
#
# for column_name in cols_to_impute:
#     df.update(pd.DataFrame(impute.knn(df, column=column_name, k=10), columns=df.keys()))

df = pd.read_csv("../Data/train-filtered-aggregated.csv")

booked_true_rows = df[df["booking_bool"]]
booked_false_rows = pd.DataFrame(columns=df.keys())
print("Total rows: " + str(len(booked_true_rows.index)))
counter = 0
for index, row in booked_true_rows.iterrows():
    srch_id = row["srch_id"]
    temp_df = df[(df["booking_bool"] == False) & (df["srch_id"] == srch_id)]
    if len(temp_df.index) > 10:
       temp_df = temp_df.head(10)
    booked_false_rows = booked_false_rows.append(temp_df)
    counter+=1
    if (counter % 1000) == 0:
        print(counter)
