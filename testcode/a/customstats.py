import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class sort_wrapper():
    """Made to easily sort things items based on a value that can be separately given."""

    def __init__(self, item, value, absolute=False):
        self.item = item
        self.value = value
        self.abs = absolute

    def __gt__(self, other):
        if self.abs:
            return abs(self.value) > (other.value)
        else:
            return self.value > other.value

    def __repr__(self):
        return f'feature: {self.item}, score: {self.value}'

    def rprint(self, dec):
        print(f'feature: {self.item}, score: {round(self.value, dec)}')


def lprint(list_, dec=None):
    """Prints the list (indented)"""

    for elem in list_:
        if dec is not None:
            elem.rprint(dec)
        else:
            print(f"  {elem}")


def unique(df, column=None, a=0.4, plot="full", d=False):
    """See unique values+occurrences and plot them"""

    if column is not None:
        values = df[column].unique()
        print(f"There are {df.nunique()[column]} unique values.")
        print(df[column].value_counts())
    # else:
        # pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        # print(df.nunique())
    if plot:
        if plot == "hist":
            df[column].hist()
            plt.title(f"Histogram of {column}")
            plt.ylabel("Occurrence")
            plt.xlabel("Value")
        if plot == "full":
            custom_hist(df[column], label="", density=d)
            plt.title(f"Histogram of {column}")
            plt.ylabel("Occurrence")
            plt.xlabel("Value")
        plt.show()
    # print(f"The {len(values)} unique values are: {values}")

def over_time(df, column, a=0.1):
    """Scatter the measurements over time"""

    plt.scatter(df.index, df[column], alpha=a)
    plt.title(f"Occurrence over time of {column}")
    plt.ylabel("Value")
    plt.xlabel("Row")
    plt.show()


#todo,

def co_occurrence(df, col1, col2, a=0.1, inv=False, scale="linear"):
    "Scatter the co occurrence of the collumns"
    if inv:
        plt.scatter(x=(df[col1] - df[col1].max()) * -1, y=df[col2], alpha=a)
    else:
        plt.scatter(x=df[col1], y=df[col2], alpha=a)
    plt.title(f'Co occurance of {col1} and {col2}')
    plt.xscale(scale)
    plt.xlabel(col1)
    plt.ylabel(col2)

def check_correlation(df, col1, col2=None):
    cor = df.corr()
    if col2:
        print(f"The correlation between {col1} and {col2}: {cor[col1][col2]:0.4}")
        print(f"Stats for {col1} correlations with all columns:\nmean: {cor[col1].drop(labels=[col1]).mean()}\nstd: {cor[col1].drop(labels=[col1]).std()}\nhigh: {cor[col1].drop(labels=[col1]).max()}\nlow: {cor[col1].drop(labels=[col1]).min()}\n")
        print(f"Stats for {col2} correlations with all columns:\nmean: {cor[col2].drop(labels=[col1]).mean()}\nstd: {cor[col2].drop(labels=[col2]).std()}\nhigh: {cor[col2].drop(labels=[col2]).max()}\nlow: {cor[col2].drop(labels=[col2]).min()}")
    else:
        print(f"Stats for {col1} correlations with all columns:\nmean: {cor[col1].drop(labels=[col1]).mean()}\nstd: {cor[col1].drop(labels=[col1]).std()}\nhigh: {cor[col1].drop(labels=[col1]).max()}\nlow: {cor[col1].drop(labels=[col1]).min()}")

def best_correlations(df, col, top=5, absolute=True, dec=4):
    print(f"==== Best correlations for {col} ====")
    cor = df.corr()
    list_ = []
    for c in cor.drop(labels=[col]):
        if not c == col:
            list_.append(sort_wrapper(c, cor[col][c], absolute))
    low = sorted(list_)
    high = low[::-1]
    print('High values:')
    lprint(high[:top], dec=dec)
    print('\nLow values:')
    lprint(low[:top], dec=dec)

def count_nans(df, fraction=True, dec=3):
    nancount = df.isna().sum()
    rows = len(df.index)
    if fraction:
        print(f"Fraction of NaNs in the first {rows} rows:\n\n")
        print((nancount/rows).round(dec).replace(0, "0"))
    else:
        print(f"Out of {rows} rows:\n\n")
        print(nancount)

def conditional_plot(df, col1, col2):
    """Plots histograms of col2 given col1"""
    plt.title(f"Occurrences of {col1} given {col2}")
    for v in df[col1].unique():
        df[df[col1] == v][col2].hist(label=f"{col1} = {v}", alpha=0.4)
    plt.legend()
    plt.show()

def rando_conditional(df, col1, col2, value_to_check=None, tight=True, a=0.4, d=True):
    """checks if random changes the relation between col1 and col2"""

    not_rand = df[df["random_bool"]==0]
    rand = df[df["random_bool"]==1]

    plt.title(f"Occurances of {col1} given {col2} when randomized")
    if value_to_check is None:
        for v in rand[col1].unique():
            rand[rand[col1] == v][col2].hist(label=f"{col1} = {v}", alpha=0.4)
        plt.legend()
        plt.show()

    plt.title(f"Occurances of {col1} given {col2} when ordered")
    if value_to_check is None:
        for v in not_rand[col1].unique():
            not_rand[not_rand[col1] == v][col2].hist(label=f"{col1} = {v}", a=0.4)
        plt.legend()
        plt.show()

    if value_to_check is not None:
        if value_to_check is not None:
            if tight:
                custom_hist(rand[rand[col1] == value_to_check][col2], label="random_bool=1", a=a, density=d)
                custom_hist(not_rand[not_rand[col1] == value_to_check][col2], label=f"random_bool=0", a=a, density=d)
            else:
                rand[rand[col1] == value_to_check][col2].hist(label="random_bool=1", alpha=0.4, density=True)
                not_rand[not_rand[col1] == value_to_check][col2].hist(label=f"random_bool=0", alpha=0.4, density=True)
        plt.xlabel(col2)
        plt.ylabel(f"count (density) of {col1} = {value_to_check}")
        plt.legend()
        plt.show()

# TODO
def custom_hist(series, label="", a=0.9, density=True, scale="linear"):
    count = {x: 0 for x in series}
    print(series)
    for x in series:
        count[x] += 1
    y = np.array([count[x] for x in sorted(series)])
    if density:
        y = y/len(series)
    x = [x for x in sorted(series)]
    # print(y)
    # print(x)
    # input()
    plt.yscale(scale)
    plt.plot(x, y, label=label, alpha=a)



def check_occurrence_per_query(df, col, value=None, rate=True, dec=4):
    total = {val: 0 for val in df[col].unique()}
    counter = 0

    for i in df["srch_id"].unique():
        subdf = df[df["srch_id"] == i]
        vals = subdf[col].unique()
        for val in vals:
            total[val] += 1
        counter += 1

    if rate:
        total = {val: total[val]/counter for val in df[col].unique()}

    if value is not None:
        print(f"The value {col}={value} occurs with a probability of {total[val]}")
    else:
        for val in total:
            print(f"The value {col}={val} occurs with a probability of {round(total[val], 4)}")

# TODO Hoevaak all nans binnen een srch_id (property distance & )
# def check_nans_per_query(df, col, value=None, rate=True, dec=4):
#     all_nans = 0
#     none_nans = 0
#     rest = 0
#     for i in df["srch_id"].unique():
#         all = 1
#         none = 1
#         subdf = df[df["srch_id"] == i]
#
#         for row in subdf.iterrows():
#             if np.isnan(row[1][col]):
#                 if all == 1:
#                     all = 0
#             else:
#                 if none == 1:
#                     none = 0
#         all_nans += all
#         none_nans += none
#         rest += 1 - all - none
#     if rate:
#         total = all_nans + none_nans + rest
#         all_nans, none_nans, rest = all_nans/total, none_nans/total, rest/total
#     print(f"All nans: {all_nans}\nNo nans: {none_nans}\nMixed: {rest}")

def check_nans_per_query(df, col, rate=True, dec=3):
    nancount = df.isna()
    nancount["srch_id"] = df["srch_id"]
    nancount.groupby(["srch_id"]).mean()
    all_nans = 0
    none_nans = 0
    rest = 0
    for row, val in nancount.iterrows():
        if val[col] == 1:
            all_nans += 1
        if val[col] == 0:
            none_nans += 1
        else:
            rest += 1
        # input(val[col])
    if rate:
        total = all_nans + none_nans + rest
        all_nans, none_nans, rest = all_nans/total, none_nans/total, rest/total
    print(f"All nans: {all_nans}\nNo nans: {none_nans}\nMixed: {rest}")



def whiten_per_query(df, col):
    whitened = []
    for i in df["srch_id"].unique():
        subdf = df[df["srch_id"] == i]
        unwhitened = []

        for row in subdf.iterrows():
            unwhitened.append(row[1][col])

        unwhitened = np.array(unwhitened)
        unwhitened -= unwhitened.mean()
        unwhitened /= unwhitened.std()
        for w in unwhitened:
            whitened.append(w)
    df[f'whitened {col}'] = whitened

def norm_per_query(df, col):
    whitened = []
    for i in df["srch_id"].unique():
        subdf = df[df["srch_id"] == i]
        unwhitened = []

        for row in subdf.iterrows():
            unwhitened.append(row[1][col])

        unwhitened = np.array(unwhitened)
        unwhitened -= unwhitened.min()
        unwhitened /= unwhitened.max()
        for w in unwhitened:
            whitened.append(w)
    df[f'normalized {col}'] = whitened

def probability_of_property_given_random():
    pass


#PROPERTY BASED CHECK
