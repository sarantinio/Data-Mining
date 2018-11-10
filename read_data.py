import csv
from datetime import datetime
import sPickle as sPickle
import _pickle as cPickle
import sys
import numpy as np
import pandas as pd

def init_dict(isTraining):
    new_dict = {}
    new_dict['srch_id'] = None
    new_dict['date_time'] = None
    new_dict['site_id'] = None
    new_dict['visitor_location_country_id'] = None
    new_dict['visitor_hist_starrating'] = None
    new_dict['visitor_hist_adr_usd'] = None
    new_dict['prop_country_id'] = None
    new_dict['prop_id'] = None
    new_dict['prop_starrating'] = None
    new_dict['prop_review_score'] = None
    new_dict['prop_brand_bool'] = None
    new_dict['prop_location_score1'] = None
    new_dict['prop_location_score2'] = None
    new_dict['prop_log_historical_price'] = None
    new_dict['price_usd'] = None
    new_dict['promotion_flag'] = None
    new_dict['srch_destination_id'] = None
    new_dict['srch_length_of_stay'] = None
    new_dict['srch_booking_window'] = None
    new_dict['srch_adults_count'] = None
    new_dict['srch_children_count'] = None
    new_dict['srch_room_count'] = None
    new_dict['srch_saturday_night_bool'] = None
    new_dict['srch_query_affinity_score'] = None
    new_dict['orig_destination_distance'] = None
    new_dict['random_bool'] = None
    new_dict['comp1_rate'] = None
    new_dict['comp1_inv'] = None
    new_dict['comp1_rate_percent_diff'] = None
    new_dict['comp2_rate'] = None
    new_dict['comp2_inv'] = None
    new_dict['comp2_rate_percent_diff'] = None
    new_dict['comp3_rate'] = None
    new_dict['comp3_inv'] = None
    new_dict['comp3_rate_percent_diff'] = None
    new_dict['comp4_rate'] = None
    new_dict['comp4_inv'] = None
    new_dict['comp4_rate_percent_diff'] = None
    new_dict['comp5_rate'] = None
    new_dict['comp5_inv'] = None
    new_dict['comp5_rate_percent_diff'] = None
    new_dict['comp6_rate'] = None
    new_dict['comp6_inv'] = None
    new_dict['comp6_rate_percent_diff'] = None
    new_dict['comp7_rate'] = None
    new_dict['comp7_inv'] = None
    new_dict['comp7_rate_percent_diff'] = None
    new_dict['comp8_rate'] = None
    new_dict['comp8_inv'] = None
    new_dict['comp8_rate_percent_diff'] = None
    if isTraining:
        new_dict['position'] = None
        new_dict['click_bool'] = None
        new_dict['booking_bool'] = None
        new_dict['gross_bookings_usd'] = None
    return new_dict


def convert_data(filenameload, filenamesave, isTraining):
    values_arr = []
    with open(filenameload) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            new_dict = init_dict(isTraining)
            for key, value in row.items():
                new_dict[key] = change_data(key, value)
            values_arr.append(new_dict)
        df = pd.DataFrame(values_arr)
        df.to_csv(filenamesave)



def change_data(key, originalValue):
    if key == "srch_id" or key == "site_id" or key == "visitor_location_country_id" \
            or key == "prop_country_id" or key == "prop_id" or key == "prop_starrating" \
            or key == "position" or key == "srch_destination_id" or key == "srch_length_of_stay" \
            or key == "srch_booking_window" or key == "srch_booking_window" or key == "srch_adults_count" \
            or key == "srch_children_count" or key == "srch_room_count" or key == "":
        return int(originalValue)
    elif key == "date_time":
        return datetime.strptime(originalValue, '%Y-%m-%d %H:%M:%S')
    elif key == "visitor_hist_starrating" or key == "visitor_hist_adr_usd" or key == "prop_location_score2" \
            or key == "srch_query_affinity_score" or key == "orig_destination_distance" or key == "gross_bookings_usd" \
            or key == "comp1_rate_percent_diff" or key == "comp2_rate_percent_diff" or key == "comp3_rate_percent_diff" \
            or key == "comp4_rate_percent_diff" or key == "comp5_rate_percent_diff" or key == "comp6_rate_percent_diff" \
            or key == "comp7_rate_percent_diff" or key == "comp8_rate_percent_diff":
        return None if originalValue == 'NULL' else float(originalValue)
    elif key == "prop_review_score":
        return None if (originalValue == 'NULL' or originalValue == '') else float(originalValue)
    elif key == "prop_brand_bool" or key == "promotion_flag" or key == "srch_saturday_night_bool" \
            or key == "random_bool" or key == "click_bool" or key == "booking_bool" \
            or key == "comp1_inv" or key == "comp2_inv" or key == "comp3_inv" \
            or key == "comp4_inv" or key == "comp5_inv" or key == "comp6_inv" \
            or key == "comp7_inv" or key == "comp8_inv":
        return True if originalValue == '1' else False if originalValue == '0' else None
    elif key == "comp1_rate" or key == "comp2_rate" or key == "comp3_rate" \
            or key == "comp4_rate" or key == "comp5_rate" or key == "comp6_rate" \
            or key == "comp7_rate" or key == "comp8_rate":
        return None if originalValue == 'NULL' else int(originalValue)
    elif key == "prop_location_score1" or key == "prop_log_historical_price" or key == "price_usd":
        return float(originalValue)


def load_data_pickle(filename):
    f = open(filename, 'rb')
    # unpickler = cPickle.Unpickler(f)
    return cPickle.load(f)

def load_data_csv(filename, delimiter):
    with open(filename) as csvfile:
        return csv.reader(csvfile, delimiter=delimiter)

def count_nones(data):
    for key, value in data.items():
        print(key + " number of None: " + str(sum(x is None for x in data[key])) + " out of " + str(len(data[key])))

def aggregate_comp(data):
    comp_inv, comp_per_diff, comp_rate = extract_all_comp_values(data)

    comp_inv_joined, comp_per_diff_joined, comp_rate_joined = join_comp_arrays(comp_inv, comp_per_diff, comp_rate)

    print(len(comp_inv_joined))
    print("Number of None: " + str(sum(np.isnan(x) for x in comp_inv_joined)) + " out of " + str(len(comp_inv_joined)))
    print("Number of None: " + str(sum(np.isnan(x) for x in comp_rate_joined)) + " out of " + str(len(comp_inv_joined)))
    print("Number of None: " + str(sum(np.isnan(x) for x in comp_per_diff_joined)) + " out of " + str(len(comp_inv_joined)))

    data = add_joined_data(data, comp_inv_joined, comp_per_diff_joined, comp_rate_joined)
    data = remove_all_comp_values(data)
    return data


def join_comp_arrays(comp_inv, comp_per_diff, comp_rate):
    comp_inv_joined = np.zeros(len(comp_inv[0]))
    comp_per_diff_joined = np.zeros(len(comp_per_diff[0]))
    comp_rate_joined = np.zeros(len(comp_rate[0]))

    for x in range(len(comp_inv[0])):
        new_inv, new_diff, new_rate = get_new_comp_values(comp_inv, comp_per_diff, comp_rate, x)
        assign_min_value(comp_inv_joined, comp_per_diff_joined, comp_rate_joined, new_inv, new_diff, new_rate, x)

    return comp_inv_joined, comp_per_diff_joined, comp_rate_joined


def assign_min_value(comp_inv_joined, comp_per_diff_joined, comp_rate_joined, min_comp_inv, min_per_diff, min_rate, x):
    comp_inv_joined[x] = min_comp_inv
    comp_per_diff_joined[x] = min_per_diff
    comp_rate_joined[x] = min_rate


def get_new_comp_values(comp_inv, comp_per_diff, comp_rate, x):
    inv_data = []
    diff_data = []
    rate_data = []
    for y in range(len(comp_inv)):
        if (not np.isnan(comp_inv[y][x])):
            inv_data.append(comp_inv[y][x])
        if (not np.isnan(comp_per_diff[y][x])):
            diff_data.append(comp_per_diff[y][x])
        if (not np.isnan(comp_rate[y][x])):
            rate_data.append(comp_rate[y][x])
    if len(inv_data) != 0:
        return_inv = most_common(inv_data, False)
    else:
        return_inv = None
    if len(diff_data) != 0:
        return_diff = most_common(diff_data, False)
    else:
        return_diff = None
    if len(rate_data) != 0:
        return_rate = most_common(rate_data,True)
    else:
        return_rate = None
    return return_inv, return_diff, return_rate

def most_common(lst, rate):
    if rate:
        return float(sum(lst)) / max(len(lst), 1)
    else:
        return max(set(lst), key=lst.count)

def extract_all_comp_values(data):
    comp_inv = []
    comp_per_diff = []
    comp_rate = []
    for key, value in data.items():
        if 'comp' in key and 'inv' in key:
            comp_inv.append(value)
        elif 'comp' in key and 'rate_percent_diff' in key:
            comp_per_diff.append(value)
        elif 'comp' in key and 'rate' in key:
            comp_rate.append(value)
    return comp_inv, comp_per_diff, comp_rate


def add_joined_data(data, comp_inv_joined, comp_per_diff_joined, comp_rate_joined):
    data['comp_inv'] = comp_inv_joined
    data['comp_rate'] = comp_rate_joined
    data['comp_rate_percent_diff'] = comp_per_diff_joined
    return data


def remove_all_comp_values(data):
    data = remove_comp_values(data, '1')
    data = remove_comp_values(data, '2')
    data = remove_comp_values(data, '3')
    data = remove_comp_values(data, '4')
    data = remove_comp_values(data, '5')
    data = remove_comp_values(data, '6')
    data = remove_comp_values(data, '7')
    data = remove_comp_values(data, '8')
    return data


def remove_comp_values(data, number):
    data.pop('comp'+number+'_rate')
    data.pop('comp'+number+'_inv')
    data.pop('comp'+number+'_rate_percent_diff')
    return data

def filter_dates(data):
    month = []
    day = []
    for value in data['date_time']:
        if type(value) == str:
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        month.append(value.month)
        day.append(value.day)
    data['month'] = month
    data['day'] = day
    data.pop('date_time')
    return data

if (__name__ == "__main__"):
    convert_data("../Data/test.csv", '../Data/test-updated.csv', False)
    data= pd.read_csv("../Data/test-updated.csv")
    print("filtering")
    data = filter_dates(data)
    print("aggregating")
    data = aggregate_comp(data)
    data.to_csv("../Data/filtered-aggregated-test.csv")



