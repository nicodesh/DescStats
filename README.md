# DescStats
A Python module to realize Descriptive Statistics: Univariate Analysis and Bivariate Analysis

## Examples

First example  here: https://github.com/nicodesh/rester-livre/blob/master/rester-livre.ipynb
Note that in the linked project, the code of the different classes is directly in the project, so it's not the up-to-date module code. But it's quite enough to understand how the module works.

Second exemple here: https://github.com/nicodesh/market-research-with-desc-stats

I used PCA and hierarchical clustering to group countries based on FAO data (Found and Agriculture Organization).

## Description
There are five classes:

## BotPlotStyle:
It's just a class with some properties for pyplot boxplots.

## MyPlot:
A class to facilitate plot creations with matplotlib.pyplot, like:
- Pie Chart
- Scatter Plot
- Linear Regression
- Unique Boxplot
- Multiple Boxtplots

## Univa:
A class to realize univariate analysis. You can describe your variable, plot the distribution, a bar chart, a pie chart, a Lorenz Curve... You can also analyze a variable depending on a filter, which has to be another variable with the same length.

## Biva:
A class to realize bivariate analysis. You can realize different kind of bivariate analysis, depending on the nature of each variable. You can calculate:

- Pearson (R²)
- Covariance (σxy)
- Eta Squared (η2)
- Khi2 (χ²)

You can plot boxplots for ANOVA, heatmap for Khi2, linear regression and boxplots for ANOVA with two quantitative variables.

## MyPCA
Compute and plot PCA analyses: scatter plot, correlation circle and so on.

## MyHier
Compute and plot hierarchical clustering, dendrogram, centroïds.
