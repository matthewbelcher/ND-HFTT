import csv
from datetime import datetime, timedelta
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import re 
from tqdm import tqdm

futures_months = {
    1 : "F",
    2 : "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z"
}

def get_futures_code(release_date: datetime) -> str:
    return futures_months[release_date.month].lower() + str(release_date.year)[-2:]


def get_ff_price(futures_code:str, time: datetime) -> float | None:
    if not os.path.exists(f'data/federal_funds/combined/ff{futures_code}.csv'):
        return None
    df = pd.read_csv(f'data/federal_funds/combined/ff{futures_code}.csv', parse_dates=['Time'])

    if  len(df[df['Time'] > time]) == 0:
        return None
    
    return df.iloc[df[df['Time'] > time].iloc[0].name - 1]['Trade']

def get_ff_price_on_day(futures_code:str, day: datetime) -> float | None:
    if not os.path.exists(f'data/federal_funds/formatted_daily/ff{futures_code}.csv'):
        print(f"File not found for futures code: {futures_code}")
        return None
    df = pd.read_csv(f'data/federal_funds/formatted_daily/ff{futures_code}.csv', parse_dates=['date'])
    
    if  len(df[df['date'] <= day]) == 0:
        print(f"No data found for date: {day} in file: ff{futures_code}.csv")
        return None
    price =  df.iloc[df[df['date'] <= day].iloc[-1].name]['price']
    return price

def get_rate(time: datetime) -> tuple[float,float]:
    df = pd.read_csv('data/target_rates/target_rates_3.csv', parse_dates=['date'])
    last_targets = df.iloc[df[df['date'] >= time].iloc[0].name - 1]
    return float(last_targets['target_low']), float(last_targets['target_high'])

def get_days_in_month(date:datetime) -> int:
    first_day = date.replace(day=1)

    if date.month == 12:
        next_month = datetime(year=date.year + 1, month=1, day=1)
    else:
        next_month = datetime(year=date.year, month=date.month + 1, day=1)
    return (next_month - first_day).days

def get_months_till_no_meeting(release_dates: list[datetime], current_date: datetime) -> list[datetime]:
    months_till_no_meeting = []
    next_no_meeting_month = current_date.month 
    next_no_meeting_year = current_date.year
    
    while True:
        is_meeting = False
        for date in release_dates:
            if date.month == next_no_meeting_month and date.year == next_no_meeting_year:
                is_meeting = True
                months_till_no_meeting.append(date)
                if next_no_meeting_month >= 12:
                    next_no_meeting_month = 1
                    next_no_meeting_year += 1
                else:
                    next_no_meeting_month += 1
                break
        if not is_meeting:
            if next_no_meeting_month > current_date.month or next_no_meeting_year > current_date.year:
                months_till_no_meeting.append(datetime(year=next_no_meeting_year, month=next_no_meeting_month, day=1))
                return months_till_no_meeting
            else:
                months_till_no_meeting.append(datetime(year=next_no_meeting_year, month=next_no_meeting_month, day=1))
                next_no_meeting_month += 1
                if next_no_meeting_month > 12:
                    next_no_meeting_month = 1
                    next_no_meeting_year += 1

def get_release_date(release_dates: list[datetime], month: int, year: int) -> int | None:
    for date in release_dates:
        if date.month == month and date.year == year:
            return date.day

    return None

def get_path_to_anchor_month(release_dates: list[datetime], relative_date) -> list["MonthlyEFFR"]:
    months_till_no_meeting = get_months_till_no_meeting(release_dates, relative_date)
    monthly_effrs = []
    for date in months_till_no_meeting:
        monthly_effrs.append(MonthlyEFFR(date.month, date.year, relative_date))
    return monthly_effrs    

class MonthlyEFFR:
    def __init__(self, month: int, year: int, relative_date: datetime):
        self.month = month
        self.year = year

        self.release_date = get_release_date(fomc_release_dates, month, year)
        self.has_release = self.release_date is not None

        self.days_in_month = get_days_in_month(datetime(year=self.year, month=self.month, day=1))
        if self.has_release:
            self.days_before_release = self.release_date
            self.days_after_release = self.days_in_month - self.release_date
        
        self.futures_price = get_ff_price_on_day(get_futures_code(datetime(year=self.year, month=self.month, day=1)), relative_date)
        self.implied_rate = (100 - self.futures_price) if self.futures_price is not None else None

        if self.has_release:
            self.start_rate = None
            self.end_rate = None
        else:
            self.start_rate = self.implied_rate
            self.end_rate = self.implied_rate
        
        self.rate_hike_probabilities = dict()
        self.cumulative_rate_hike_probabilities = dict()
    


class EFFRAnchorPath:
    def __init__(self, relative_date: datetime):
        self.relative_date = relative_date
        self.monthly_effrs = get_path_to_anchor_month(fomc_release_dates, relative_date)
        self.path_length = len(self.monthly_effrs)
    
    def calculate_probabilities(self):
        self.__connect_start_end_rates()

        for index in range(len(self.monthly_effrs)):
            current_month = self.monthly_effrs[index]
            if current_month.has_release:
                effr_delta = current_month.end_rate - current_month.start_rate
                partial_hikes, full_hikes = math.modf(abs(effr_delta) / 0.25)

                current_month.rate_hike_probabilities = {
                    (-1 if effr_delta < 0 else 1) * full_hikes * 0.25 + 0: 1 - partial_hikes,
                    (-1 if effr_delta < 0 else 1) * (full_hikes + 1) * 0.25 + 0 : partial_hikes
                }

        for index in range(len(self.monthly_effrs) - 1):
            current_month = self.monthly_effrs[index]
            if len(current_month.cumulative_rate_hike_probabilities) == 0:
                current_month.cumulative_rate_hike_probabilities = current_month.rate_hike_probabilities.copy()
            next_month = self.monthly_effrs[index + 1]
            if next_month.has_release:
                for current_rate_change, current_prob in current_month.cumulative_rate_hike_probabilities.items():
                    for next_rate_change, next_prob in next_month.rate_hike_probabilities.items():
                        combined_rate_change = next_rate_change + current_rate_change
                        if combined_rate_change not in next_month.cumulative_rate_hike_probabilities:
                            next_month.cumulative_rate_hike_probabilities[combined_rate_change] = 0
                        next_month.cumulative_rate_hike_probabilities[combined_rate_change] += next_prob * current_prob

    def print_path(self):
        for effr in self.monthly_effrs:
            print(f"Month: {effr.month}, Year: {effr.year}, Number of Days: {effr.days_in_month}, Futures Code: {get_futures_code(datetime(year=effr.year, month=effr.month, day =1))}, Futures Price: {effr.futures_price},Start Rate: {effr.start_rate}, Implied Rate: {effr.implied_rate}, End Rate: {effr.end_rate}, Rate Hike Probabilities: {effr.cumulative_rate_hike_probabilities}")
            print()
            
    def __connect_start_end_rates(self):
        for index in range(len(self.monthly_effrs) - 1, -1, -1):
            current_month = self.monthly_effrs[index]
            if current_month.has_release:
                if current_month.end_rate is not None and current_month.start_rate is None:
                    # print(f"Connecting start and end rates for month: {current_month.month}, year: {current_month.year}")
                    current_month.start_rate = (current_month.implied_rate - ((current_month.days_after_release/current_month.days_in_month) * current_month.end_rate )) / (current_month.days_before_release/current_month.days_in_month)
            
            if index >= 1:
                self.monthly_effrs[index - 1].end_rate = current_month.start_rate
        
        if self.path_length > 1 and not self.monthly_effrs[0].has_release:
            self.monthly_effrs[1].start_rate = self.monthly_effrs[0].implied_rate

fomc_release_dates = sorted([re.search(r'monetary(.*)a.htm', x).group(1) for x in os.listdir('data/fomc_statements')])
fomc_release_dates = [datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])) for date in fomc_release_dates]

fomc_release_dates = list(filter(lambda x: x >= datetime(year=2008, month=12, day=16), fomc_release_dates))

print(f"Total FOMC release dates: {len(fomc_release_dates)}")
print(fomc_release_dates)
for date in tqdm(reversed(fomc_release_dates[:-1]), leave=True):
    print(f"Processing date: {date.strftime('%m-%d-%Y')}")
    historical_prob = []
    for days_prior in tqdm(range(0, 120), leave=True):
        try:
            relative_date = date - timedelta(days=days_prior)
            anchor_path = EFFRAnchorPath(relative_date)
            anchor_path.calculate_probabilities()
            for effr in anchor_path.monthly_effrs:
                if effr.month == date.month and effr.year == date.year:
                    row = []
                    row.append(relative_date.strftime("%m-%d-%Y"))
                    for change in range(-200, 201, 25):
                        row.append(effr.cumulative_rate_hike_probabilities.get(change/100, 0))
                    if not any(row[1:]):
                        print(effr.cumulative_rate_hike_probabilities)
                        anchor_path.print_path()
                        continue
                        exit()
                    historical_prob.append(row)

        except TypeError as e:
            print(f"Processed {days_prior - 1} days for date : {date.strftime('%m-%d-%Y')}")
            print(f"Failed on {relative_date.strftime('%m-%d-%Y')} for {get_futures_code(date)}")
            break

    historical_prob.sort(key=lambda x: datetime.strptime(x[0], "%m-%d-%Y"))
    
    df = pd.DataFrame(historical_prob, columns=["Date", "-200bp", "-175bp", "-150bp", "-125bp", "-100bp", "-75bp", "-50bp", "-25bp", "0bp", "25bp", "50bp", "75bp", "100bp", "125bp", "150bp", "175bp", "200bp"])
    df.to_csv(f"data/federal_funds/historical_probabilities_3/{date.strftime('%m_%d_%Y')}.csv", index=False)
    print()

# date = datetime(year=2024, month=10, day=2)
# print(f"Processing date: {date.strftime('%m-%d-%Y')}")
# historical_prob = []
# for days_prior in range(0, 90):
#     print(f"Processing days prior: {days_prior}")
#     try:
#         relative_date = date - timedelta(days=days_prior)
#         print(f"Relative date: {relative_date.strftime('%m-%d-%Y')}")
#         anchor_path = EFFRAnchorPath(relative_date)
#         for effr in anchor_path.monthly_effrs:
#             print(f"Month: {effr.month}, Year: {effr.year}, Start Rate: {effr.start_rate}, Implied Rate: {effr.implied_rate}, End Rate: {effr.end_rate}")
#         anchor_path.calculate_probabilities()
#         for effr in anchor_path.monthly_effrs:
#             if effr.month == relative_date.month and effr.year == relative_date.year:
#                 historical_prob.append([relative_date.strftime("%m-%d-%Y"), effr.cumulative_rate_hike_probabilities.get(-0.5, 0), effr.cumulative_rate_hike_probabilities.get(-0.25, 0), effr.cumulative_rate_hike_probabilities.get(0), effr.cumulative_rate_hike_probabilities.get(0.25, 0), effr.cumulative_rate_hike_probabilities.get(0.5, 0)])
    
#     except TypeError as e:
#         print(e)
#         print(f"Processed {days_prior - 1} days for date : {date.strftime('%m-%d-%Y')}")
#         break

# historical_prob.sort(key=lambda x: datetime.strptime(x[0], "%m-%d-%Y"))

# df = pd.DataFrame(historical_prob, columns=["Date", "-50bp", "-25bp", "0bp", "25bp", "50bp"])
# df.to_csv(f"data/federal_funds/historical_probabilities/{date.strftime('%m_%d_%Y')}.csv", index=False)

# anchor_path = EFFRAnchorPath(datetime(year=2024, month=10, day=2))
# anchor_path.calculate_probabilities()
# anchor_path.print_path()

# fomc_release_dates += [datetime(year=2025, month=5, day=7), datetime(year=2025, month=6, day=18), datetime(year=2025, month=7, day=30), datetime(year=2025, month=9, day=17)]
# relative_date = datetime(year=2025, month=1, day=1)
# anchor_path = EFFRAnchorPath(datetime(year=2025, month=1, day=12))

# anchor_path.monthly_effrs[0].implied_rate = 100 - 95.6875	
# anchor_path.monthly_effrs[1].implied_rate = 100 - 95.7450
# anchor_path.monthly_effrs[2].implied_rate = 100 - 95.8350
# anchor_path.monthly_effrs[3].implied_rate = 100 - 95.9150
# anchor_path.monthly_effrs[4].implied_rate = 100 - 96.1500
# anchor_path.monthly_effrs[4].start_rate = 100 -96.1500
# anchor_path.monthly_effrs[4].end_rate = 100 - 96.1500

# anchor_path.calculate_probabilities()
# anchor_path.print_path()
