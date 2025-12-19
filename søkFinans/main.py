import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Data_logreturns.csv')

data = pd.read_csv(data_path)

# a)
daily_std = data.std()
annual_std = daily_std * np.sqrt(250)

print("\na) Annualized standard deviations for each stock:")
for stock, std in annual_std.items():
    print(f"{stock}: {std:.4f}")

# b)
portfolio_stds = []
for i in range(1, len(data.columns) + 1):
    portfolio = data.iloc[:, :i].mean(axis=1)
    daily_std = portfolio.std()
    annual_std = daily_std * np.sqrt(250)
    portfolio_stds.append(annual_std)


# c)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), portfolio_stds, 'bo-')
plt.xlabel('Number of Stocks in Portfolio')
plt.ylabel('Annualized Standard Deviation')
plt.title('Portfolio Diversification Effect')
plt.grid(True)
plt.show()

# e)
num_simulations = 1000
random_portfolio_stds = []

for size in range(1, len(data.columns) + 1):
    size_stds = []
    for _ in range(num_simulations):
        selected_cols = np.random.choice(
            data.columns, size=size, replace=False)
        portfolio = data[selected_cols].mean(axis=1)
        annual_std = portfolio.std() * np.sqrt(250)
        size_stds.append(annual_std)
    random_portfolio_stds.append(np.mean(size_stds))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), random_portfolio_stds, 'ro-')
plt.xlabel('Number of Stocks in Portfolio')
plt.ylabel('Annualized Standard Deviation')
plt.title('Portfolio Diversification Effect (Random Sampling)')
plt.grid(True)
plt.show()


# 2)
def bond_price(par, coupon_rate, periods, yield_rate):
    """
    Calculate bond price with semi-annual payments
    par: face value
    coupon_rate: annual coupon rate
    periods: number of semi-annual periods
    yield_rate: annual yield rate
    """
    semi_annual_yield = (1 + yield_rate) ** 0.5 - \
        1
    semi_annual_coupon = (coupon_rate * par) / 2

    pv = 0
    for t in range(1, periods + 1):
        pv += semi_annual_coupon / ((1 + semi_annual_yield) ** t)

    pv += par / ((1 + semi_annual_yield) ** periods)
    return pv


# a)
par_value = 450_000_000
issue_price = 0.98 * par_value
print(f"\nMoney raised: ${issue_price:,.2f}")

# b)
coupon_rate = 0.135
semi_annual_coupon = (coupon_rate * par_value) / 2
print(f"Semi-annual coupon payment: ${semi_annual_coupon:,.2f}")

# c)
yield_rate = 0.1467
periods = 8  # 4 years * 2 payments per year
calculated_price = bond_price(par_value, coupon_rate, periods, yield_rate)
actual_price = issue_price

print(f"Calculated price at 14.67% yield: ${calculated_price:,.2f}")
print(f"Actual issue price: ${actual_price:,.2f}")

# Printouts:
# Task 1
# a) Annualized standard deviations for each stock:
#     Ford: 0.3546
#     Clorox: 0.2444
#     Chubb: 0.2032
#     Hormel Food: 0.2201
#     Grainger: 0.2537
#     AT & T: 0.2704
#     AbbVie: 0.1967
#     Amcor: 0.2168
#     Ecolab: 0.2050
#     Pentair: 0.2792

# Task 2
# a) Money raised: $441, 000, 000.00
# b) Semi-annual coupon payment: $30, 375, 000.00
# c) Calculated price at 14.67 % yield: $441, 052, 188.79
#    Actual issue price: $441, 000, 000.00
