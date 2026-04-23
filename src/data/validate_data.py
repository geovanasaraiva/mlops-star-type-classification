from scipy.stats import ks_2samp, chi2_contingency

def compare_distributions(train_df, test_df, columns):
    #Dictionary that stores the results of statistical tests.
    results = {}

    #Iterates through each column that we want to compare between train and test.
    for col in columns:
        #Remove missing values
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()

        #If the column is numeric, the Kolmogorov-Smirnov rule applies.
        if train_df[col].dtype in ["int64", "float64"]:
            #KS test. Compares continuous distributions between two groups.
            stat, p = ks_2samp(train_vals, test_vals)
            results[col] = {"test": "KS", "statistic": stat, "p_value": p}
        
        else: #For categorical variables, use Chi²

            #Account for the proportion of each category in the train
            train_counts = train_vals.value_counts(normalize=True)
            test_counts = test_vals.value_counts(normalize=True)

            #Combine all existing categories
            all_cats = sorted(set(train_counts.index).union(test_counts.index))

            #Creates aligned probability vectors for each category
            train_probs = [train_counts.get(c, 0) for c in all_cats]
            test_probs = [test_counts.get(c, 0) for c in all_cats]

            #Chi² test for comparing categorical distributions
            chi2, p, _, _ = chi2_contingency([train_probs, test_probs])

            #Stores Chi² test result
            results[col] = {"test": "Chi2", "statistic": chi2, "p_value": p}

    return results