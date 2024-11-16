import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

user_rates_file = 'User Rates.xlsx'
market_rates_file = 'Market Row Data.xlsx'


def closest_user_closest_percentile(market_prices, user_price   ):
    percentiles = list(range(1, 101))
    # We basically calculate potential saving for each percentile, and then we choose the min value as the closest
    closest_percentile = min(percentiles, key=lambda p: abs(np.percentile(market_prices, p) - user_price))
    return closest_percentile, np.percentile(market_prices, closest_percentile) - user_price


def calculate_closest_user_percentile_price(user_data, market_data):
    result = []
    for index, row in user_data.iterrows():
        filtered_market = market_data[
            (market_data['origin'] == row['origin']) & (market_data['destination'] == row['destination'])
        ]
        if filtered_market.empty:
            continue
        ordered_market_price = sorted(filtered_market['price'].values)
        closest_percentile, closest_percentile_value  = closest_user_closest_percentile(ordered_market_price, row['price'])
        result.append({
            **row,
            'closest_percentile': closest_percentile,
            'closest_percentile_value': closest_percentile_value,
        })
    return result

def predict_future_prices(market_data):
    future_prices_predictions = []
    market_data['date_numeric'] = pd.to_datetime(market_data['date']).map(datetime.datetime.toordinal)
    routes = market_data[['origin', 'destination']].drop_duplicates()
    for _, route in routes.iterrows():
        route_data = market_data[
            (market_data['origin'] == route['origin']) & (market_data['destination'] == route['destination'])
        ]
        x = route_data[['date_numeric']]
        y = route_data[['price']]
        model = LinearRegression()
        model.fit(x.values, y)

        future_dates = [datetime.datetime.now() + datetime.timedelta(days=day) for day in range(1, 31)]
        future_dates_numeric = [date.toordinal() for date in future_dates]
        future_prices = model.predict(np.array(future_dates_numeric).reshape(-1, 1))

        for date, price in zip(future_dates, future_prices):
            future_prices_predictions.append({
                'origin': route['origin'],
                'destination': route['destination'],
                'future_date': date,
                'future_price': price
            })
    return future_prices_predictions



def main():
    user_rates_data = pd.read_excel(user_rates_file, sheet_name='Sheet1')
    market_rates_data = pd.read_excel(market_rates_file, sheet_name='market_row_data')
    user_closest_percentile_data = calculate_closest_user_percentile_price(user_rates_data, market_rates_data)
    user_closest_percentile_data_df = pd.DataFrame(user_closest_percentile_data)
    user_closest_percentile_data_df.to_excel('user_closest_percentile_data.xlsx', index=False)
    future_prediction = predict_future_prices(market_rates_data)
    future_prediction_df = pd.DataFrame(future_prediction)
    future_prediction_df.to_excel('future_prediction.xlsx', index=False)


if __name__ == '__main__':
    main()