import _pickle as cPickle
from statistics import mean
import json
import pandas as pd
import numpy as np
import csv
import sys

def load_data(filename):
    f = open(filename, 'rb')
    return cPickle.load(f)


def init_dict(isTraining):
    new_dict = {}
    new_dict['srch_id'] = []
    #new_dict['date_time'] = []
    new_dict['day'] = []
    new_dict['month'] = []
    new_dict['site_id'] = []
    new_dict['visitor_location_country_id'] = []
    new_dict['visitor_hist_starrating'] = []
    new_dict['visitor_hist_adr_usd'] = []
    new_dict['prop_country_id'] = []
    new_dict['prop_id'] = []
    new_dict['prop_starrating'] = []
    new_dict['prop_review_score'] = []
    new_dict['prop_brand_bool'] = []
    new_dict['prop_location_score1'] = []
    new_dict['prop_location_score2'] = []
    new_dict['prop_log_historical_price'] = []
    new_dict['price_usd'] = []
    new_dict['promotion_flag'] = []
    new_dict['srch_destination_id'] = []
    new_dict['srch_length_of_stay'] = []
    new_dict['srch_booking_window'] = []
    new_dict['srch_adults_count'] = []
    new_dict['srch_children_count'] = []
    new_dict['srch_room_count'] = []
    new_dict['srch_saturday_night_bool'] = []
    new_dict['srch_query_affinity_score'] = []
    new_dict['orig_destination_distance'] = []
    new_dict['random_bool'] = []
    new_dict['comp_rate'] = []
    new_dict['comp_inv'] = []
    new_dict['comp_rate_percent_diff'] = []
    if isTraining:
        new_dict['position'] = []
        new_dict['click_bool'] = []
        new_dict['booking_bool'] = []
        new_dict['gross_bookings_usd'] = []
    return new_dict

def clean_none_values(list):
    return [x is not None for x in list]

def change_data_format(data):
    new_dict = init_dict(True)
    counter = 0
    for row in data:
        for key, value in row.items():
            if key == '' or key == 'Unnamed: 0' or key == 'Unnamed: 1':
                continue
            new_dict[key].append(value)
    return new_dict

def statistics(path):
    with open(path) as csvfile:
        analysis_dict = []
        data = csv.DictReader(csvfile)
        data = change_data_format(data)
        for key,value in data.items():
            average = 0
            maximum=0
            minimum = 0
            total_v_num = len(value)
            missing_v_num = sum(x is '' for x in value)
            filled_v_num = sum(x is not '' for x in value)
            missing_v_prct = round((missing_v_num/total_v_num) * 100,2)
            filled_v_prct = round(100-missing_v_prct,2)

            if missing_v_prct != float(100) and filled_v_prct != float(100) and "bool" not in key:
                cleaned_list = clean_none_values(value)
                average = round(mean(cleaned_list),2)
                minimum = round(min(cleaned_list),2)
                maximum = round(max(cleaned_list),2)

            analysis_dict.append({"missing_data_num":missing_v_num,"missing_data_percentage":missing_v_prct,\
                                  "filled_data_num":filled_v_num, "filled_data_percentage": filled_v_prct, "name": key,\
                                  "avg":average, "max":maximum,"min":minimum})

    with open('statistics.json', 'w') as outfile:
        json.dump(analysis_dict, outfile)

    return analysis_dict

def correlations_training_data(path):
    columns = ["srch_room_count", "visitor_location_country_id", "srch_booking_window", \
               "click_bool", "prop_country_id", "date_time", "booking_bool", "prop_starrating", \
               "srch_length_of_stay", "prop_location_score1", "position", "random_bool", \
               "srch_children_count", "srch_saturday_night_bool", "srch_adults_count", \
               "prop_log_historical_price", "price_usd", "srch_id", "prop_brand_bool", \
               "promotion_flag", "srch_destination_id", "site_id", "prop_id"]

    df = pd.read_csv(path, header=0, usecols=columns)
    correlations = df.corr()
    # plt.matshow(correlations)
    # plt.show()
    return correlations

def load_csv_pandas(path):
    return pd.read_csv(path, header=0)

def load_csv(path):
    with open(path, newline='') as csvfile:
        return csv.DictReader(csvfile)

def predictions(data):
    return data
if (__name__ == "__main__"):
    path = sys.argv[1]
    #data = load_csv("../Data/train-filtered-aggregated.csv")
    #print(type(data))
    statistics(path)
    with open('statistics.json') as json_data:
        d = json.load(json_data)
        for index in d:
            print("")
            if index["filled_data_percentage"] != float(0):
                print(index["name"]+": " + str(index["missing_data_percentage"]))

    #data = load_dataframe_from_csv("../train-filtered-aggregated.csv")

    #data = load_data('training.pkl')
    #analyzed_data = statistic(data)