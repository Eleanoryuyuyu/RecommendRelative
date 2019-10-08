import os
import logging
from collections import defaultdict, Counter
import boto3
import pandas as pd
from sqlalchemy import create_engine
import pyfpgrowth


DB_URL = "postgresql://"


def read_sql(sql):
    engine = create_engine(
        DB_URL,
        echo=True,
        pool_recycle=3600,
        pool_pre_ping=True)

    df = pd.read_sql(sql, engine)
    return df


def df2transaction_list(input_df,key,nearestK=1000000):
    def drop_rep(x):
        x = x.split(',')
        x = list(set(x))
        return x

    input_df["create_date"] = pd.to_datetime(input_df["create_date"])
    input_df = input_df.set_index("create_date").sort_index(ascending=False)
    input_df["category_list"] = input_df["category_list"].apply(lambda x: drop_rep(x))
    input_df = input_df[input_df["category_list"].apply(lambda x: 10 > len(x) > 1)]
    if key =="week":
        data = input_df["category_list"].iloc[:nearestK].values.tolist()
    else:
        data = input_df["category_list"].values.tolist()
    return data


def generate_C2C_rules(data_list, support_ratio=0.00003, confidence_ratio=0.3):
    support = support_ratio * len(data_list)
    patterns = pyfpgrowth.find_frequent_patterns(data_list, support)
    rules = pyfpgrowth.generate_association_rules(patterns, confidence_ratio)
    rules_counter = Counter()
    new_rules_t = defaultdict(lambda: 0)
    new_rules = defaultdict(lambda : dict())

    for pattern, r in rules.items():
        # target_c = r[0][0]
        confidence = r[1]
        for triggerC in pattern:
            for targetC in r[0]:
                new_rules_t[(triggerC, targetC)] += confidence
                rules_counter.update([(triggerC, targetC)])

    new_rules_t = {k: v / rules_counter[k] for k, v in new_rules_t.items()}
    for k, v in new_rules_t.items():
        c1, c2 = k
        new_rules[c1].update({c2: v})

    return new_rules


def save_to_S3_by_boto3(c2c, data_full_path):
    # print('data_path :{}'.format(data_full_path))
    df = pd.DataFrame(list(c2c.items()), columns=["category", "association"])
    df.set_index("category")
    json_data = df.to_json(orient='records', lines=True)
    full_path_split = data_full_path.split('/')
    full_path_split = list(filter(None, full_path_split))
    bucket = full_path_split[1]
    data_path = os.path.join(*full_path_split[2:])
    # print(bucket, data_path)
    s3_resource = boto3.resource('s3')
    if len(c2c) > 1:
        ret = s3_resource.Object(bucket, data_path).put(
        Body=json_data)
        logging.info('save to {}'.format(data_path))


def run_week_C2C_task(save_path):
    print("run_week_C2C_task")
    sql_week = """
           select user_id,create_date,order_week,category_list
           from test.yjy_category_list_half_year
    """
    df = read_sql(sql_week)
    print("df len {}".format(len(df)))
    data_list = df2transaction_list(df,"week")
    print("data_list len {}".format(len(data_list)))
    rules = generate_C2C_rules(data_list)
    print("df size:{}, transaction size: {}, rules size {}".format(len(df), len(data_list), len(rules)))
    save_to_S3_by_boto3(rules, save_path)


def run_order_C2C_task(save_path):
    print("run_order_C2C_task")
    sql_order = """
      select user_id,order_name,create_date,order_week,category_list
      from test.yjy_category_list_half_year2
    """
    df = read_sql(sql_order)
    print("df len {}".format(len(df)))
    data_list = df2transaction_list(df,"order")
    print("data_list len {}".format(len(data_list)))
    rules = generate_C2C_rules(data_list)
    print("df size:{}, transaction size: {}, rules size {}".format(len(df), len(data_list), len(rules)))
    save_to_S3_by_boto3(rules, save_path)


def main():
    week_path = "s3://"
    order_path = "s3://"
    run_week_C2C_task(week_path)
    run_order_C2C_task(order_path)


if __name__ == '__main__':
    main()
