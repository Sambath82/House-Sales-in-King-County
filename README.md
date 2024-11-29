# 🏠 House Price Prediction: Empowering Real Estate Decisions

## Objective
This project aims to predict house prices with high accuracy, achieving a **Mean Absolute Error (MAE) below $20,000**, using historical data and key property features. The goal is to provide **buyers**, **sellers**, and **real estate professionals** with actionable insights for making informed decisions in a competitive market.

## Dataset

This project uses the **House Sales in King County, USA** dataset available on Kaggle. The dataset provides detailed information on home sales in King County, Washington, including property features, sale prices, and geographical data.

### Dataset Source

- **Kaggle URL**: [House Sales in King County Dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

### Features Description

The dataset includes the following variables:

| **Variable**    | **Description**                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------ |
| `id`            | A unique identifier for a house                                                                  |
| `date`          | The date the house was sold                                                                      |
| `price`         | Sale price of the house (target variable for prediction)                                         |
| `bedrooms`      | Number of bedrooms                                                                               |
| `bathrooms`     | Number of bathrooms                                                                              |
| `sqft_living`   | Square footage of the living space                                                               |
| `sqft_lot`      | Square footage of the lot                                                                        |
| `floors`        | Total number of floors (levels) in the house                                                     |
| `waterfront`    | Indicator if the house has a view of the waterfront                                              |
| `view`          | Number of times the property was viewed                                                          |
| `condition`     | Overall condition rating of the house                                                            |
| `grade`         | Overall grade given to the house, based on the King County grading system                        |
| `sqft_above`    | Square footage of the house excluding the basement                                               |
| `sqft_basement` | Square footage of the basement                                                                   |
| `yr_built`      | Year the house was originally built                                                              |
| `yr_renovated`  | Year the house was last renovated                                                                |
| `zipcode`       | ZIP code of the location                                                                         |
| `lat`           | Latitude coordinate                                                                              |
| `long`          | Longitude coordinate                                                                             |
| `sqft_living15` | Square footage of interior living space in 2015 (may reflect renovations affecting living space) |
| `sqft_lot15`    | Square footage of the lot in 2015 (may reflect renovations affecting lot size)                   |

### Target Variable
The target variable for this project is **`price`**, which represents the sale price of a house. 






