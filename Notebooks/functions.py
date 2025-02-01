import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def summarize_dataframe(df):
    """
    Summarizes the given DataFrame by displaying the total number of data types, 
    counts of unique values for float, integer, and object types, and the DataFrame's dimensions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be summarized.
        
    Returns:
    --------
    None
    """
    print("\nData Total Number of Each Type:\n", df.dtypes.value_counts())
    
    print("\nFloat Types Count:\n", df.select_dtypes('float64').apply(pd.Series.nunique, axis=0))
    
    print("\nInteger Types Count:\n", df.select_dtypes('int64').apply(pd.Series.nunique, axis=0))
    
    print("\nObject Types Count:\n", df.select_dtypes('object').apply(pd.Series.nunique, axis=0))
    
    print("\nData Dimension:", df.shape)


def missing_values_table(df):
    """
    Calculates and displays a table of missing values in the DataFrame.

    This function identifies missing values in each column of the DataFrame,
    calculates the percentage of missing values, and presents the results
    in a structured format.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame for which to calculate missing values.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the number and percentage of missing values 
        for each column with missing data.

    Notes:
    ------
    - The output DataFrame is sorted by the percentage of missing values in 
      descending order.
    - Only columns with missing values are included in the output.
    """

    mis_val = df.isnull().sum()
    
    mis_val_percent = 100 * mis_val / len(df)
    
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    mis_val_table.columns = ['Missing Values', '% of Total Values']
    
    mis_val_table = mis_val_table[mis_val_table['% of Total Values'] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    print(f"Number of columns of this dataframe is {df.shape[1]}.\n"
          f"Columns with missing values are {mis_val_table.shape[0]}.")
    
    return mis_val_table


def handle_missing_values(data):
    """
    This function handles missing values in a DataFrame.
    - First, it checks and prints columns with missing values by data type.
    - Replaces missing values:
        - Continuous data: replaces with mean.
        - Categorical data: replaces with the most frequent value.
    - Finally, checks and prints the columns with missing values after imputation.
    - Returns the cleaned DataFrame.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    
    print("Initial Missing Values by Column:")
    missing_before = data.isnull().sum()
    for col in data.columns:
        if missing_before[col] > 0:
            print(f"Column: {col}, Missing Values: {missing_before[col]}, Type: {data[col].dtype}")

    # Handle missing values
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in [np.float64, np.int64]:
                # Continuous data: replace missing values with mean
                mean_value = data[col].mean()
                data[col].fillna(mean_value, inplace=True)
            else:
                # Categorical data: replace missing values with the most frequent value
                mode_value = data[col].mode()[0]
                data[col].fillna(mode_value, inplace=True)

    print("\nMissing Values by Column After Handling:")
    missing_after = data.isnull().sum()
    for col in data.columns:
        if missing_after[col] > 0:
            print(f"Column: {col}, Missing Values: {missing_after[col]}")
        else:
            print(f"Column: {col} has no missing values.")

    return data

def convert_to_categorical(df, min_unique=2, max_unique=30):
    """
    Converts columns of type int64 with a unique value count within a specified range 
    to categorical type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to modify.
    min_unique : int, optional
        The minimum number of unique values a column must have to be converted (default is 2).
    max_unique : int, optional
        The maximum number of unique values a column must have to be converted (default is 10).

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with specified columns converted to categorical type.
    """
    # Identify columns of type int64 with unique values between min_unique and max_unique
    cols_to_convert = [
        col for col in df.select_dtypes(['float64', 'int64', 'int32', 'object', 'category']).columns 
        #col for col in df.select_dtypes(['object']).columns 
        if min_unique <= df[col].nunique() <= max_unique
    ]

    df[cols_to_convert] = df[cols_to_convert].astype('category')

    print(f"Converted columns to categorical: {cols_to_convert}")
    
    return df

def encode_and_one_hot(df):
    """
    Encodes categorical columns with 2 or fewer unique values using label encoding,
    and applies one-hot encoding to the remaining categorical columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to encode.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with label encoding applied to columns with 2 or fewer unique values
        and one-hot encoding applied to all other categorical columns.
    int
        The count of columns that were label encoded.

    Notes:
    ------
    - Columns with 2 or fewer unique values are label encoded, which replaces each unique 
      category with a numeric code.
    - Columns with more than 2 unique values are one-hot encoded, which creates binary columns 
      for each unique category.
    - This function prints the shape of the DataFrame before and after encoding and displays
      the names of columns that were encoded.
    """

    print('DataFrame shape before encoding:', df.shape)
    
    le = LabelEncoder()
    le_count = 0
    label_encoded_columns = []  

    for col in df.select_dtypes(include=['object', 'category']).columns:
        if len(df[col].unique()) <= 2:
            df[col] = le.fit_transform(df[col])
            le_count += 1  
            label_encoded_columns.append(col)

    if label_encoded_columns:
        print(f"Label-encoded columns: {label_encoded_columns}")
    else:
        print("No columns were label-encoded.")

    # Get the remaining categorical columns for one-hot encoding
    one_hot_encoded_columns = list(
        set(df.select_dtypes(include=['object', 'category']).columns) - set(label_encoded_columns)
    )

    # Apply one-hot encoding to the remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    if one_hot_encoded_columns:
        print(f"One-hot encoded columns: {one_hot_encoded_columns}")
    else:
        print("No columns were one-hot encoded.")

    print('DataFrame shape after encoding:', df.shape)
    
    return df


def convert_to_integer(df, min_unique=2, max_unique=30):
    """
    Converts columns with a unique value count within a specified range 
    to integer type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to modify.
    min_unique : int, optional
        The minimum number of unique values a column must have to be converted (default is 2).
    max_unique : int, optional
        The maximum number of unique values a column must have to be converted (default is 30).

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with specified columns converted to integer type.
    """
    
    cols_to_convert = [
        col for col in df.select_dtypes(['float64', 'object']).columns 
        #col for col in df.select_dtypes(['object']).columns 
        if min_unique <= df[col].nunique() <= max_unique
    ]

    df[cols_to_convert] = df[cols_to_convert].astype('int64')

    print(f"Converted columns to integer: {cols_to_convert}")
    
    return df


def plot_boxplot_category(df, categorical_column, target_column, title="Boxplot", x_label="Category", y_label="Target"):
    """
    Plots a boxplot of a target variable against a categorical feature.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    categorical_column : str
        The name of the categorical column to group by.
    target_column : str
        The name of the continuous target variable.
    title : str, optional
        The title of the plot (default is "Boxplot").
    x_label : str, optional
        The label for the x-axis (default is "Category").
    y_label : str, optional
        The label for the y-axis (default is "Target").
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=categorical_column, y=target_column, data=df, palette="Set2")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def boxplot_and_convert_bin(df, column_name, num_bins=3):
    """
    Create a histogram, bin the specified column into bins based on unique values,
    visualize the distribution of bins, and map the binned categories to numeric values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column to bin and analyze.
    num_bins : int
        The number of bins to create.

    Returns:
    --------
    pandas.DataFrame
        The updated DataFrame with the new binned column and numeric mapping added.
    """

    plt.figure(figsize=(8, 4))
    plt.hist(df[column_name], bins=10, color='skyblue', edgecolor='black')
    plt.xlabel(column_name.capitalize())
    plt.ylabel("Count")
    plt.title(f"Histogram of {column_name.capitalize()}")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # Calculate bin intervals based on unique values
    unique_values = sorted(df[column_name].unique())
    num_unique = len(unique_values)
    bin_size = max(1, num_unique // num_bins)  # Ensure at least 1 unique value per bin
    bins = [unique_values[i] for i in range(0, num_unique, bin_size)]
    if bins[-1] != unique_values[-1]:  # Ensure the last value is included in the last bin
        bins.append(unique_values[-1])

    # Generate bin labels
    group_names = [f"Bin {i+1}" for i in range(len(bins) - 1)]
    binned_column_name = f'{column_name}-binned'

    # Bin the column
    df[binned_column_name] = pd.cut(df[column_name], bins=bins, labels=group_names, include_lowest=True)

    # Map the binned categories to numeric values
    bin_numeric_map = {group_names[i]: i + 1 for i in range(len(group_names))}
    numeric_column_name = f'{column_name}-binned-numeric'
    df[numeric_column_name] = df[binned_column_name].map(bin_numeric_map)

    print(df[[column_name, binned_column_name, numeric_column_name]].head(10))

    print("\nValue counts for binned:")
    print(df[binned_column_name].value_counts())

    plt.figure(figsize=(8, 4))
    df[binned_column_name].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.xlabel(f"{column_name.capitalize()} Binned")
    plt.ylabel("Count")
    plt.title(f"Histogram of {column_name.capitalize()} Binned")
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(rotation=0)
    plt.show()

    return df



def detect_outliers_iqr_all(df, target, factor=1.5):
    """
    Detects outliers in all continuous features of the DataFrame
    using the IQR method and visualizes them with box plots.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    factor : float, optional
        The multiplier for the IQR range (default is 1.5 for mild outliers).

    Returns:
    --------
    list
        A list of column names that contain outliers.
    """
    
    continuous_columns = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target]).columns
    outlier_columns = []  

    num_cols = len(continuous_columns)
    fig, axes = plt.subplots(nrows=(num_cols + 2) // 3, ncols=3, figsize=(15, num_cols * 1.5))
    fig.suptitle("Outlier Detection Using Box Plots", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, column in enumerate(continuous_columns):
        # Calculate Q1, Q3, and IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the outlier bounds
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        num_outliers = outliers.sum()

        # Print information about the outliers
        print(f"Outliers detected in '{column}': {num_outliers} rows.")
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}\n")

        # If there are outliers, add the column to the list
        if num_outliers > 0:
            outlier_columns.append(column)

        # Plot the box plot with outliers
        ax = axes[idx]
        ax.boxplot(df[column].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
        ax.set_title(f'{column} (Outliers: {num_outliers})')
        ax.set_xlabel(column)

        # Highlight outliers in red
        outlier_values = df.loc[outliers, column]
        ax.scatter(outlier_values, np.ones_like(outlier_values), color='red', label='Outliers', alpha=0.7)

    # Remove any empty subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return outlier_columns 


def preprocess_data(df, target, factor=1.5):
    """
    Preprocess the DataFrame by detecting outliers using the IQR method 
    and replacing them with the mean values of their respective columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to preprocess.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers replaced by column mean values.
    """
   
    # Detect numeric columns with outliers
    numeric_outlier_columns = detect_outliers_iqr_all(df, target, factor)

    # Replace outliers in numeric columns with the mean values
    for column in numeric_outlier_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mean_value = df[column].mean()

        # Replace outliers with the mean
        df[column] = df[column].apply(
            lambda x: mean_value if x < lower_bound or x > upper_bound else x
        )

    return df

def transform_to_log(df, features):
    """
    Apply log transformation to specified features in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    features : list of str
        The list of column names to transform.
    
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with log-transformed features.
    """
    df_transformed = df.copy()

    for feature in features:
        # Apply log transformation based on feature values
        log_feature_name = f"log_{feature}"
        df_transformed[log_feature_name] = np.where(
            df_transformed[feature] == 0,
            np.log(df_transformed[feature] + 1),
            np.log(df_transformed[feature])
        )
    
    return df_transformed



def plot_corr_heatmap(df):
    """
    Plots a heatmap of correlations for all features in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the features.
    
    Returns:
    --------
    None
    """
    
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, 
                cbar=True, square=True, linewidths=0.5, alpha=0.8)
    
    plt.title('Correlation Heatmap')
    plt.show()


def evaluate_model(model, X_train, X_test, Y_train, Y_test, is_log_transformed=True):
    """
    Fits the given model on X and Y, makes predictions, 
    and evaluates the model using R² and MSE metrics.

    Parameters:
    -----------
    model : sklearn estimator
        The regression model to evaluate.
    X : pandas.DataFrame or numpy.ndarray
        The feature set for training the model.
    Y : pandas.Series or numpy.ndarray
        The target variable.

    Returns:
    --------
    None
         Prints the R² score, RMSE, and MAE of the model.
    """
  
    model.fit(X_train, Y_train)

    Y_pred_log = model.predict(X_test)
    if is_log_transformed:
        # Reverse log transformation
        Y_pred = np.exp(Y_pred_log)
        Y_true = np.exp(Y_test)
    else:
        Y_pred = Y_pred_log
        Y_true = Y_test

    r2_score_value = r2_score(Y_true, Y_pred)
    mse = mean_squared_error(Y_true, Y_pred)
    mae = mean_absolute_error(Y_true, Y_pred)
    mape = mean_absolute_percentage_error(Y_true, Y_pred)
    rmse = np.sqrt(mse)

    print(f"R-squared Score: {r2_score_value:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 10
    height = 8
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Houses')
    plt.legend()
    plt.show()
    plt.close()

def plot_histogram(data, column, bins=30, title=None, xlabel=None, ylabel='Frequency', color='skyblue', edge_color='black'):
    """
    Plots a histogram with a transparent background, custom color, and labels.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the column to plot.
    
    Returns:
    --------
    None
        Displays the histogram with customized styling and labels.
    """
    # Set up figure 
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)  

    # Plot histogram
    ax.hist(data[column], bins=bins, color=color, edgecolor=edge_color)

    # Set title and labels
    ax.set_title(f'Histogram of {title}' if title  else f'Histogram of {column}', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel if xlabel else column, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Style grid and layout
    ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5)
    plt.tight_layout()  

    # Show plot
    plt.show()