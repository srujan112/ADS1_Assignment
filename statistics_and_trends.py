import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as ss


def plot_relational_plot(df):
  """Generates a scatter plot showing the relationship between carat and price."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=df['carat'], y=df['price'], alpha=0.5
    )
    plt.xlabel('Carat')
    plt.ylabel('Price')
    plt.title('Relationship Between Carat and Price')
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    """Generates a bar plot showing the average price by cut quality."""
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x='cut', y='price', data=df,
        estimator=lambda x: sum(x) / len(x)
    )
    plt.xlabel('Cut Quality')
    plt.ylabel('Average Price')
    plt.title('Average Price by Cut Quality')
    plt.savefig('categorical_plot.png')
    plt.show()


def plot_statistical_plot(df):
    """Generates a heatmap to show correlations between numerical variables."""
    plt.figure(figsize=(8, 5))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(
        numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f'
    )
    plt.title('Correlation Heatmap')
    plt.savefig('statistical_plot.png')
    plt.show()


def statistical_analysis(df, col: str):
    """Calculates mean, standard deviation, skewness, and excess kurtosis."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Performs basic preprocessing including checking data structure."""
    print(f"Summary Stats:\n{df.describe()}\n")
    numeric_df = df.select_dtypes(include=['number'])
    print(f"Correlation Matrix:\n{numeric_df.corr()}\n")
    print(f"First Rows:\n{df.head()}\n")
    df.dropna(inplace=True)
    return df


def writing(moments, col):
    """Prints the statistical moments and provides interpretation."""
    print(f'For the attribute {col}:')
    print(
        f'Mean = {moments[0]:.2f}, Std Dev = {moments[1]:.2f}, '
        f'Skewness = {moments[2]:.2f},\n'
        f'Excess Kurtosis = {moments[3]:.2f}.'
    )
    skewness_desc = (
        "right-skewed" if moments[2] > 0 else "left-skewed"
        if moments[2] < 0 else "symmetrical"
    )
    kurtosis_desc = (
        "leptokurtic" if moments[3] > 0 else "platykurtic"
        if moments[3] < 0 else "mesokurtic"
    )
    print(f'The data is {skewness_desc} and {kurtosis_desc}.')


def main():
    """Main function to execute data analysis tasks."""
    df = pd.read_csv('Diamonds Prices2022.csv')
    df = preprocessing(df)
    col = 'price'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
