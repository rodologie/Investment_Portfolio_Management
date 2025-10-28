### Sources for IPM TA
import statistics as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_excess_return(rf, beta, rm):
    """
    Calculate the excess return of an asset using the Capital Asset Pricing Model (CAPM).

    Parameters:
    rf (float): Risk-free rate
    beta (float): Beta of the asset
    rm (float): Expected market return

    Returns:
    float: Excess return of the asset
    """
    excess_return = rf + beta * (rm - rf)
    return excess_return

def calculate_alpha(expected_return, rf, beta, rm):
    """
    Calculate the alpha of an asset.

    Parameters:
    expected_return (float): Expected return of the asset
    rf (float): Risk-free rate
    beta (float): Beta of the asset
    rm (float): Expected market return

    Returns:
    float: Alpha of the asset
"""


    capm_return = calculate_excess_return(rf, beta, rm)
    alpha = expected_return - capm_return
    return alpha


def calculate_variance_per_asset(investment_amount, std_dev=0.3):
    """
    Calculate the variance per asset.

    Parameters:
    investment_amount (float): Amount invested in the asset
    std_dev (float): Standard deviation of the asset's returns

    Returns:
    float: Variance per asset
    """
    variance = (investment_amount * std_dev) ** 2
    return variance


def return_CAPM(alpha, beta, Rm, ei):
    """
    Calculate the return of an asset using the Capital Asset Pricing Model (CAPM).

    Parameters:
    alpha (float): Alpha of the asset
    beta (float): Beta of the asset
    Rm (float): Market return
    ei (float): Firm-specific return

    Returns:
    float: Return of the asset
    """
    Ri = alpha + beta * Rm + ei
    return Ri


def calculate_formule_variance(alpha, beta, sigma_M, sigma_ei):
    """
    Calculate the variance of returns for an asset.

    Parameters:
    alpha (float): Alpha of the asset
    beta (float): Beta of the asset
    Rm (float): Market return
    ei (float): Firm-specific return

    Returns:
    float: Variance of returns for the asset
    """
    var = (beta ** 2) * sigma_M**2 + sigma_ei**2  # because cov(Rm, ei) = 0
    return var



# def print_sml(df_tmp):

#     a, b = np.polyfit(df_tmp['beta'], df_tmp['Expected return'], 1)
#     x = np.linspace(df_tmp['beta'].min() - 0.1, df_tmp['beta'].max() + 0.1, 200)
#     y_fit = a * x + b


#     eq = f"y = {a:.4f} x + {b:.4f}"
#     plt.figure(figsize=(8, 6))


#     plt.title('CAPM: Expected Return vs Beta')
#     plt.xlabel('Beta')
#     plt.ylabel('Expected Return')
#     plt.scatter(df_tmp['beta'], df_tmp['Expected return'], color='blue')
#     plt.scatter(df_tmp['beta'], df_tmp['forecast_return'],
#                 color='green', marker='x', label='Forecast Return')
#     plt.plot(x, y_fit, color='red', label='CAPM Line')

#     for idx in df_tmp.index:
#         beta = float(df_tmp.loc[idx, 'beta'])
#         exp_ret = float(df_tmp.loc[idx, 'Expected return'])
#         plt.scatter(beta, exp_ret, color='blue', s=60, zorder=3)
#         plt.annotate(f"{idx} ({beta:.2f}, {exp_ret:.2f})",
#                     (beta, exp_ret),
#                     xytext=(6, -6), textcoords='offset points',
#                     color='blue', fontsize=9)

#         if 'forecast_return' in df_tmp.columns and not pd.isna(df_tmp.loc[idx, 'forecast_return']):
#             f_ret = df_tmp.loc[idx, 'forecast_return']
#             x_proj = beta #(beta + a * (f_ret - b)) / (1 + a**2)
#             y_proj = a * x_proj + b

#             plt.plot([beta, x_proj], [f_ret, y_proj],
#                     linestyle='--', color='green', linewidth=1)

#             plt.scatter([x_proj], [y_proj], color='green',
#                         s=40, zorder=5, marker='x')
#             plt.annotate(f"proj {idx}\n({x_proj:.2f}, {y_proj:.2f})",
#                         (x_proj, y_proj),
#                         xytext=(6, 6), textcoords='offset points',
#                         color='green', fontsize=8)

#         plt.text(0.05, 0.95, eq, transform=plt.gca().transAxes,
#                 fontsize=12, verticalalignment='top',
#                 bbox=dict(facecolor='white', alpha=0.8))



def print_sml(df_tmp):
    # work on a copy and ensure numeric
    df = df_tmp.copy()
    df['beta'] = pd.to_numeric(df['beta'], errors='coerce')
    df['Expected return'] = pd.to_numeric(
        df['Expected return'], errors='coerce')
    has_forecast = 'forecast_return' in df.columns
    if has_forecast:
        df['forecast_return'] = pd.to_numeric(
            df['forecast_return'], errors='coerce')

    # fit CAPM line on (beta, Expected return)
    a, b = np.polyfit(df['beta'].dropna(), df['Expected return'].dropna(), 1)
    x = np.linspace(df['beta'].min() - 0.1, df['beta'].max() + 0.1, 200)
    y_fit = a * x + b
    eq = f"y = {a:.4f} x + {b:.4f}"

    plt.figure(figsize=(8, 6))
    plt.title('CAPM: Expected Return vs Beta')
    plt.xlabel('Beta')
    plt.ylabel('Expected Return')

    # plot expected returns
    plt.scatter(df['beta'], df['Expected return'], color='blue',
                s=60, zorder=3, label='Expected return')

    # plot forecast returns if available
    if has_forecast and df['forecast_return'].notna().any():
        plt.scatter(df['beta'], df['forecast_return'], color='green',
                    marker='x', s=90, zorder=4, label='Forecast return')

    # plot CAPM line
    plt.plot(x, y_fit, color='red', label='CAPM Line', zorder=1)

    # annotate points and, if forecast exists, draw projection lines
    for idx in df.index:
        beta = float(df.loc[idx, 'beta'])
        exp_ret = float(df.loc[idx, 'Expected return'])
        # expected point
        plt.scatter(beta, exp_ret, color='blue', s=60, zorder=5)
        plt.annotate(f"{idx}\n({beta:.2f}, {exp_ret:.2f})",
                     (beta, exp_ret),
                     xytext=(6, -6), textcoords='offset points',
                     color='blue', fontsize=9)

        # forecast and projection
        if has_forecast and not pd.isna(df.loc[idx, 'forecast_return']):
            f_ret = float(df.loc[idx, 'forecast_return'])
            # original forecast point (keep initial coordinate)
            plt.scatter(beta, f_ret, color='green', marker='x', s=90, zorder=6)
            # plt.annotate(f"orig {idx}\n({beta:.2f}, {f_ret:.2f})",
            #              (beta, f_ret),
            #              xytext=(6, 6), textcoords='offset points',
            #              color='green', fontsize=9)

            # orthogonal projection of (beta, f_ret) onto y = a x + b
            x_proj = beta #(beta + a * (f_ret - b)) / (1 + a**2)
            y_proj = a * x_proj + b

            # dashed line from original forecast point to its projection
            plt.plot([beta, x_proj], [f_ret, y_proj], linestyle='--',
                     color='green', linewidth=1, zorder=4)

            # projection point and annotation
            plt.scatter([x_proj], [y_proj], color='green',
                        s=40, zorder=7, marker='x')
            plt.annotate(f"proj {idx}\n({x_proj:.2f}, {y_proj:.2f})",
                         (x_proj, y_proj),
                         xytext=(6, 6), textcoords='offset points',
                         color='green', fontsize=8)

    # equation, grid and legend
    plt.text(0.05, 0.95, eq, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True)
    plt.legend()
    plt.show()

