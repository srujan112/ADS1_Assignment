import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as ss


def plot_relational_plot(df):
    """Generates a relational plot (line plot showing trends over years)."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='year', y='suicides_no', hue='country', data=df)
    plt.title('Suicide Trends Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Suicides')
    plt.legend(title='Country')
    plt.grid()
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    """Generates a categorical plot (bar chart for suicide numbers by gender)."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x='sex', y='suicides_no', data=df, estimator=sum)
    plt.title('Total Suicides by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Total Suicides')
    plt.grid()
    plt.savefig('categorical_plot.png')
    plt.show()


def plot_statistical_plot(df):
    """Generates a statistical plot (heatmap showing correlation)."""
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[['suicides_no', 'population', 'year']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
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
    print(f'Mean = {moments[0]:.2f}, ')
    print(f'Standard Deviation = {moments[1]:.2f}, ')
    print(f'Skewness = {moments[2]:.2f}, and ')
    print(f'Excess Kurtosis = {moments[3]:.2f}.')
    
    skewness_desc = "not skewed" if -0.5 < moments[2] < 0.5 else (
        "right-skewed" if moments[2] > 0.5 else "left-skewed")
    kurtosis_desc = "mesokurtic" if -0.5 < moments[3] < 0.5 else (
        "leptokurtic" if moments[3] > 0.5 else "platykurtic")
    
    print(f'The data is {skewness_desc} and {kurtosis_desc}.')


def main():
    """Main function to execute data analysis tasks."""
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'suicides_no'
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
