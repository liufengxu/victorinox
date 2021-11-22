def get_date_list(begin_date, end_date):
    date_list = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start=begin_date, end=end_date))]
    return date_list

print(get_date_list('2021-11-01','2021-11-21'))
