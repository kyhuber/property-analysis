# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Property Analysis for Home Purchase
#
# This notebook analyzes property data to identify properties that fit your criteria and ranks them based on desirability and seller likelihood.
#
# ## 1. Setup and Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import re

# Enable interactive plots
%matplotlib inline

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Set display options to see more columns
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Print version info
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")

# %% [markdown]

# ## 2. Data Loading Function
#
# Define a function to load and clean the property data CSV.

# %%
def load_data(file_path, test_mode=False, nrows=None):
    """
    Load property data from CSV file with option to load a subset for testing
    
    Args:
        file_path (str): Path to the CSV file
        test_mode (bool): If True, only load a small sample for testing
        nrows (int): Number of rows to load if test_mode is True
        
    Returns:
        pandas.DataFrame: Property data
    """
    print(f"Loading data from {file_path}...")
    
    # Load the data, optionally just a sample
    if test_mode and nrows is not None:
        properties = pd.read_csv(file_path, nrows=nrows)
        print(f"TEST MODE: Loading {nrows} rows")
    else:
        properties = pd.read_csv(file_path)
    
    # Print original shape before cleaning
    print(f"Original data shape: {properties.shape}")
    
    # Quick data cleaning
    # Convert date columns to datetime if they exist
    if 'DocRcrdgDt_County' in properties.columns:
        properties['DocRcrdgDt_County'] = pd.to_datetime(
            properties['DocRcrdgDt_County'], 
            errors='coerce'
        )
    
    # Handle numeric columns
    numeric_columns = [
        'BsmtFinSqFt', 'BsmtUnFinSqFt', 'BathHalfCt', 'Bath3QtrCt', 
        'BathFullCt', 'BathTtlCt', 'BedCt', 'BldgSqFt', 'LotSqFt',
        'MktTtlVal', 'YrBlt', 'StoriesCt', 'Acres', 'SaleAmt_County',
        'TaxTtl1', 'TaxTtl2', 'TaxTtl3'
    ]
    
    for col in numeric_columns:
        if col in properties.columns:
            # Convert to numeric, errors='coerce' will convert invalid values to NaN
            properties[col] = pd.to_numeric(properties[col], errors='coerce')
    
    # Handle boolean columns
    boolean_columns = [
        'OwnerOccupiedInd', 'BareLandInd', 'InvestmentProp', 
        'BankOwnedInd', 'OwnerCorporateInd'
    ]
    
    for col in boolean_columns:
        if col in properties.columns:
            # Convert to boolean, accounting for various formats
            properties[col] = properties[col].map(
                {'TRUE': True, 'FALSE': False, True: True, False: False}
            )
    
    # Print data info after cleaning
    print(f"Loaded {len(properties)} properties")
    print(f"Columns: {properties.columns.tolist()[:5]}... (and {len(properties.columns)-5} more)")
    
    return properties

# %% [markdown]
# ## 3. Load The Data
#
# Load the property data, with an option to test on a small subset first.

# %%
# File path - MODIFY THIS for your environment
# Option 1: If the CSV is in the same directory as this notebook
file_path = "./Category_1.csv"

# Option 2: Full path
# file_path = "/full/path/to/your/data/Category_1.csv"

# Set to True to test with a small sample first
TEST_MODE = True

# Load data
properties = load_data(file_path, test_mode=TEST_MODE, nrows=100)

# %% [markdown]
# ## 4. Examine the Data
#
# Let's examine the data to understand what we're working with.

# %%
# Display a few sample rows
properties.head()

# %%
# Check data types and missing values
properties.info()

# %%
# Look at statistics for numeric columns
properties.describe()

# %%
# Check which ZIP codes are present
if 'SiteZIP' in properties.columns:
    print("ZIP code counts:")
    print(properties['SiteZIP'].value_counts())

# %%
# Check distribution of property types
if 'LandUseDsc' in properties.columns:
    print("Land use types:")
    print(properties['LandUseDsc'].value_counts().head(10))

# %% [markdown]
# ## 5. Filter Functions
#
# Define functions to filter properties based on our criteria.

# %%
def filter_category1(properties):
    """
    Apply Category 1 (Rent-Ready Properties) filters
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Filtered property data
    """
    # Print what we're looking for
    print("Filtering for Category 1 (Rent-Ready Properties) with criteria:")
    print("  - At least 2 bathrooms")
    print("  - At least 3 bedrooms")
    print("  - In ZIP codes: 98106, 98116, 98126, 98136")
    print("  - Market value <= $850,000")
    print("  - Property type: Single Family or Duplex")
    
    # Define filter criteria for Category 1
    category1_filter = (
        (properties['BathTtlCt'] >= 2) &
        (properties['BedCt'] >= 3) &
        (properties['SiteZIP'].isin(['98106', '98116', '98126', '98136'])) &
        (properties['MktTtlVal'] <= 850000) &
        (
            properties['LandUseDsc'].str.contains('Single Family', case=False, na=False) |
            properties['LandUseDsc'].str.contains('Duplex', case=False, na=False)
        )
    )
    
    # Apply filters
    cat1_properties = properties[category1_filter].copy()
    
    print(f"Found {len(cat1_properties)} properties matching Category 1 criteria out of {len(properties)} total properties")
    
    # Check count by ZIP code
    if len(cat1_properties) > 0 and 'SiteZIP' in cat1_properties.columns:
        print("\nBreakdown by ZIP code:")
        print(cat1_properties['SiteZIP'].value_counts())
    
    return cat1_properties


def filter_category2(properties):
    """
    Apply Category 2 (Conversion-Ready Properties) filters
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Filtered property data
    """
    # Print what we're looking for
    print("Filtering for Category 2 (Conversion-Ready Properties) with criteria:")
    print("  - At least 1.5 bathrooms")
    print("  - At least 3 bedrooms")
    print("  - Building size >= 1600 sq ft")
    print("  - Lot size > 5000 sq ft")
    print("  - In ZIP codes: 98106, 98116, 98126, 98136")
    print("  - Market value between $650,000-$800,000")
    print("  - Property type: Single Family")
    
    # Define filter criteria for Category 2
    category2_filter = (
        (properties['BathTtlCt'] >= 1.5) &
        (properties['BedCt'] >= 3) &
        (properties['BldgSqFt'] >= 1600) &
        (properties['LotSqFt'] > 5000) &
        (properties['SiteZIP'].isin(['98106', '98116', '98126', '98136'])) &
        (properties['MktTtlVal'] >= 650000) &
        (properties['MktTtlVal'] <= 800000) &
        (properties['LandUseDsc'].str.contains('Single Family', case=False, na=False))
    )
    
    # Apply filters
    cat2_properties = properties[category2_filter].copy()
    
    print(f"Found {len(cat2_properties)} properties matching Category 2 criteria out of {len(properties)} total properties")
    
    # Check count by ZIP code
    if len(cat2_properties) > 0 and 'SiteZIP' in cat2_properties.columns:
        print("\nBreakdown by ZIP code:")
        print(cat2_properties['SiteZIP'].value_counts())
    
    return cat2_properties


def filter_category3(properties):
    """
    Apply Category 3 (Single-Family Residences) filters
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Filtered property data
    """
    # Print what we're looking for
    print("Filtering for Category 3 (Single-Family Residences) with criteria:")
    print("  - At least 1.5 bathrooms")
    print("  - At least 2 bedrooms")
    print("  - In ZIP codes: 98106, 98116, 98126, 98136")
    print("  - Market value between $450,000-$650,000")
    print("  - Property type: Single Family")
    
    # Define filter criteria for Category 3
    category3_filter = (
        (properties['BathTtlCt'] >= 1.5) &
        (properties['BedCt'] >= 2) &
        (properties['SiteZIP'].isin(['98106', '98116', '98126', '98136'])) &
        (properties['MktTtlVal'] >= 450000) &
        (properties['MktTtlVal'] <= 650000) &
        (properties['LandUseDsc'].str.contains('Single Family', case=False, na=False))
    )
    
    # Apply filters
    cat3_properties = properties[category3_filter].copy()
    
    print(f"Found {len(cat3_properties)} properties matching Category 3 criteria out of {len(properties)} total properties")
    
    # Check count by ZIP code
    if len(cat3_properties) > 0 and 'SiteZIP' in cat3_properties.columns:
        print("\nBreakdown by ZIP code:")
        print(cat3_properties['SiteZIP'].value_counts())
    
    return cat3_properties


def calculate_zip_averages(properties):
    """
    Calculate average values by ZIP code
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Average values by ZIP code
    """
    zip_averages = properties.groupby('SiteZIP')['MktTtlVal'].mean().reset_index()
    zip_averages.rename(columns={'MktTtlVal': 'ZipAvgValue'}, inplace=True)
    
    print("ZIP code average values:")
    print(zip_averages)
    
    return zip_averages

# %% [markdown]
# ## 6. Apply Filters
#
# Let's apply the filters and see what properties match our criteria.

# %%
# Apply Category 1 filters
cat1_properties = filter_category1(properties)

# Display the first few properties that match
if len(cat1_properties) > 0:
    # Select a subset of columns to display
    display_columns = [
        'SiteAddr', 'SiteCity', 'SiteZIP', 'BathTtlCt', 'BedCt',
        'BldgSqFt', 'LotSqFt', 'YrBlt', 'MktTtlVal', 'LandUseDsc'
    ]
    
    # Make sure all requested columns exist
    display_columns = [col for col in display_columns if col in cat1_properties.columns]
    
    print("\nSample of matching Category 1 properties:")
    display(cat1_properties[display_columns].head())
else:
    print("\nNo properties match Category 1 criteria.")

# Calculate neighborhood averages
if len(cat1_properties) > 0:
    zip_averages = calculate_zip_averages(cat1_properties)
    
    # Merge with the properties dataframe
    cat1_properties = cat1_properties.merge(zip_averages, on='SiteZIP', how='left')

# %% [markdown]
# ## 7. Exploratory Data Analysis
#
# Let's visualize some aspects of the filtered properties to better understand them.

# %%
# Only run if we have matching properties
if len(cat1_properties) > 0:
    # Distribution of properties by ZIP code
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=cat1_properties, x='SiteZIP')
    plt.title('Number of Properties by ZIP Code', fontsize=16)
    plt.xlabel('ZIP Code', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Distribution of property values
    plt.figure(figsize=(12, 6))
    sns.histplot(cat1_properties['MktTtlVal'], bins=20, kde=True)
    plt.title('Distribution of Property Values', fontsize=16)
    plt.xlabel('Market Total Value ($)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Building Size vs Market Value
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=cat1_properties, x='BldgSqFt', y='MktTtlVal', hue='SiteZIP', alpha=0.7)
    plt.title('Building Size vs Market Value by ZIP Code', fontsize=16)
    plt.xlabel('Building Square Footage', fontsize=12)
    plt.ylabel('Market Value ($)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Boxplot of property values by ZIP code
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=cat1_properties, x='SiteZIP', y='MktTtlVal')
    plt.title('Property Value Distribution by ZIP Code', fontsize=16)
    plt.xlabel('ZIP Code', fontsize=12)
    plt.ylabel('Market Value ($)', fontsize=12)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Scoring Functions - Desirability
#
# Now let's define functions to score each property on desirability factors.

# %%
#========== DESIRABILITY SCORE FUNCTIONS ==========

def calculate_property_type_score(row):
    """
    Calculate property type score (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-15
    """
    if pd.notna(row['LandUseDsc']) and "Duplex" in str(row['LandUseDsc']):
        return 15
    elif (pd.notna(row['LandUseDsc']) and "Single Family" in str(row['LandUseDsc']) and 
          ((pd.notna(row['BsmtFinSqFt']) and row['BsmtFinSqFt'] > 500) or 
           (pd.notna(row['BathFullCt']) and row['BathFullCt'] >= 2))):
        return 10
    elif pd.notna(row['LandUseDsc']) and "Single Family" in str(row['LandUseDsc']):
        return 5
    else:
        return 0


def calculate_bathroom_distribution_score(row):
    """
    Calculate bathroom distribution score (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-15
    """
    bath_full = row['BathFullCt'] if pd.notna(row['BathFullCt']) else 0
    bath_3qtr = row['Bath3QtrCt'] if pd.notna(row['Bath3QtrCt']) else 0
    bath_half = row['BathHalfCt'] if pd.notna(row['BathHalfCt']) else 0
    
    if bath_full >= 2 and bath_3qtr >= 1:
        return 15
    elif bath_full == 2 and bath_half >= 1:
        return 12
    elif bath_full == 1 and bath_3qtr >= 1:
        return 10
    else:
        return 5


def calculate_building_size_score(row):
    """
    Calculate building size score (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['BldgSqFt']):
        if row['BldgSqFt'] > 2500:
            return 10
        elif row['BldgSqFt'] >= 2000:
            return 8
        elif row['BldgSqFt'] >= 1800:
            return 6
        elif row['BldgSqFt'] >= 1600:
            return 4
        else:
            return 2
    return 0


def calculate_basement_space_score(row):
    """
    Calculate basement space score (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['BsmtFinSqFt']):
        if row['BsmtFinSqFt'] > 800:
            return 10
        elif row['BsmtFinSqFt'] >= 500:
            return 8
        elif row['BsmtFinSqFt'] >= 200:
            return 5
        elif row['BsmtFinSqFt'] > 0:
            return 2
    return 0


def calculate_stories_count_score(row):
    """
    Calculate stories count score (5 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-5
    """
    bsmt_fin = row['BsmtFinSqFt'] if pd.notna(row['BsmtFinSqFt']) else 0
    bsmt_unfin = row['BsmtUnFinSqFt'] if pd.notna(row['BsmtUnFinSqFt']) else 0
    
    if pd.notna(row['StoriesCt']) and row['StoriesCt'] >= 2:
        return 5
    elif pd.notna(row['StoriesCt']) and row['StoriesCt'] == 1 and (bsmt_fin > 0 or bsmt_unfin > 0):
        return 3
    else:
        return 1


def calculate_zip_code_value_score(row):
    """
    Calculate ZIP code value score (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-15
    """
    if pd.notna(row['SiteZIP']):
        if row['SiteZIP'] == '98116':
            return 15
        elif row['SiteZIP'] == '98136':
            return 12
        elif row['SiteZIP'] == '98126':
            return 10
        elif row['SiteZIP'] == '98106':
            return 8
        else:
            return 5
    return 5


def calculate_lot_size_score(row):
    """
    Calculate lot size score (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['LotSqFt']):
        if row['LotSqFt'] > 8000:
            return 10
        elif row['LotSqFt'] >= 6000:
            return 8
        elif row['LotSqFt'] >= 5000:
            return 6
        else:
            return 4
    return 0


def calculate_condition_score(row):
    """
    Calculate condition score (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['Condition']):
        condition = str(row['Condition']).strip().lower()
        if condition == 'excellent':
            return 10
        elif condition == 'good':
            return 8
        elif condition == 'average':
            return 6
        elif condition == 'fair':
            return 4
        else:
            return 2
    return 0


def calculate_year_built_score(row):
    """
    Calculate year built score (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['YrBlt']):
        if row['YrBlt'] >= 2000:
            return 10
        elif row['YrBlt'] >= 1980:
            return 8
        elif row['YrBlt'] >= 1960:
            return 6
        elif row['YrBlt'] >= 1940:
            return 4
        else:
            return 2
    return 0

# %%
# Test these functions on a sample property to see if they work as expected
def test_desirability_scoring_functions():
    # Create a sample property for testing
    sample_property = pd.Series({
        'LandUseDsc': 'Single Family(Res Use/Zone)',
        'BsmtFinSqFt': 700,
        'BathFullCt': 2,
        'Bath3QtrCt': 1,
        'BathHalfCt': 0,
        'BldgSqFt': 2200,
        'StoriesCt': 1,
        'SiteZIP': '98126',
        'LotSqFt': 7000,
        'Condition': 'Good',
        'YrBlt': 1985
    })
    
    # Test each scoring function
    print("Testing desirability scoring functions on sample property...")
    print(f"Property Type Score: {calculate_property_type_score(sample_property)}")
    print(f"Bathroom Distribution Score: {calculate_bathroom_distribution_score(sample_property)}")
    print(f"Building Size Score: {calculate_building_size_score(sample_property)}")
    print(f"Basement Space Score: {calculate_basement_space_score(sample_property)}")
    print(f"Stories Count Score: {calculate_stories_count_score(sample_property)}")
    print(f"ZIP Code Value Score: {calculate_zip_code_value_score(sample_property)}")
    print(f"Lot Size Score: {calculate_lot_size_score(sample_property)}")
    print(f"Condition Score: {calculate_condition_score(sample_property)}")
    print(f"Year Built Score: {calculate_year_built_score(sample_property)}")
    
    # Calculate total desirability score
    total_score = (
        calculate_property_type_score(sample_property) +
        calculate_bathroom_distribution_score(sample_property) +
        calculate_building_size_score(sample_property) +
        calculate_basement_space_score(sample_property) +
        calculate_stories_count_score(sample_property) +
        calculate_zip_code_value_score(sample_property) +
        calculate_lot_size_score(sample_property) +
        calculate_condition_score(sample_property) +
        calculate_year_built_score(sample_property)
    )
    
    print(f"Total Desirability Score: {total_score}")


# Run the test to see if our scoring functions work correctly
test_desirability_scoring_functions()

# %% [markdown]
# ## 9. Calculate Scores for All Properties
#
# Apply the scoring functions to all filtered properties.

# %%
def calculate_all_scores(properties):
    """
    Calculate all individual scores for properties
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Property data with scores
    """
    # Calculate desirability scores
    print("Calculating desirability scores...")
    properties['PropertyTypeScore'] = properties.apply(calculate_property_type_score, axis=1)
    properties['BathroomDistributionScore'] = properties.apply(calculate_bathroom_distribution_score, axis=1)
    properties['BuildingSizeScore'] = properties.apply(calculate_building_size_score, axis=1)
    properties['BasementSpaceScore'] = properties.apply(calculate_basement_space_score, axis=1)
    properties['StoriesCountScore'] = properties.apply(calculate_stories_count_score, axis=1)
    properties['ZipCodeValueScore'] = properties.apply(calculate_zip_code_value_score, axis=1)
    properties['LotSizeScore'] = properties.apply(calculate_lot_size_score, axis=1)
    properties['ConditionScore'] = properties.apply(calculate_condition_score, axis=1)
    properties['YearBuiltScore'] = properties.apply(calculate_year_built_score, axis=1)
    
    return properties


def calculate_combined_scores(properties):
    """
    Calculate total and combined scores, and assign priority tiers
    
    Args:
        properties (pandas.DataFrame): Property data with individual scores
        
    Returns:
        pandas.DataFrame: Property data with total scores and priority tiers
    """
    # Calculate desirability score
    desirability_columns = [
        'PropertyTypeScore', 'BathroomDistributionScore', 'BuildingSizeScore',
        'BasementSpaceScore', 'StoriesCountScore', 'ZipCodeValueScore',
        'LotSizeScore', 'ConditionScore', 'YearBuiltScore'
    ]
    
    properties['DesirabilityScore'] = properties[desirability_columns].sum(axis=1)
    
    # Calculate seller likelihood score if we decide to implement those functions
    
    # For now, just use DesirabilityScore for ranking
    properties['CombinedScore'] = properties['DesirabilityScore']
    
    # Assign priority tiers
    def assign_tier(score):
        max_possible = 90  # 15+15+10+10+5+15+10+10+10
        if score >= max_possible * 0.8:  # 80% or higher
            return "Tier 1"
        elif score >= max_possible * 0.65:  # 65-80%
            return "Tier 2"
        elif score >= max_possible * 0.5:  # 50-65%
            return "Tier 3"
        else:  # Below 50%
            return "Tier 4"
    
    properties['PriorityTier'] = properties['CombinedScore'].apply(assign_tier)
    
    print("Score ranges for each tier:")
    print("  Tier 1: 72-90")
    print("  Tier 2: 58.5-71.9")
    print("  Tier 3: 45-58.4")
    print("  Tier 4: 0-44.9")
    
    return properties

# %%
# Apply scoring functions to Category 1 properties
if len(cat1_properties) > 0:
    # Calculate scores
    cat1_properties = calculate_all_scores(cat1_properties)
    cat1_properties = calculate_combined_scores(cat1_properties)
    
    # Display score summary
    print("\nScore summary statistics:")
    print(cat1_properties['DesirabilityScore'].describe())
    
    # Count properties by tier
    print("\nProperties by tier:")
    tier_counts = cat1_properties['PriorityTier'].value_counts().sort_index()
    print(tier_counts)
    
    # Display top 5 properties
    display_columns = [
        'SiteAddr', 'SiteCity', 'SiteZIP', 'BathTtlCt', 'BedCt',
        'BldgSqFt', 'YrBlt', 'MktTtlVal', 'DesirabilityScore', 'PriorityTier'
    ]
    
    # Make sure all requested columns exist
    display_columns = [col for col in display_columns if col in cat1_properties.columns]
    
    print("\nTop 5 properties by score:")
    top_properties = cat1_properties.sort_values('DesirabilityScore', ascending=False)
    display(top_properties[display_columns].head())

# %% [markdown]
# ## 10. Visualize Scored Properties
#
# Create visualizations of scored properties to better understand the distribution.

# %%
# Only create visualizations if we have properties with scores
if len(cat1_properties) > 0 and 'DesirabilityScore' in cat1_properties.columns:
    # Distribution of desirability scores
    plt.figure(figsize=(12, 6))
    sns.histplot(cat1_properties['DesirabilityScore'], bins=20, kde=True)
    plt.title('Distribution of Desirability Scores', fontsize=16)
    plt.xlabel('Desirability Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add vertical lines for tier thresholds
    plt.axvline(x=72, color='green', linestyle='--', label='Tier 1 Threshold (72)')
    plt.axvline(x=58.5, color='blue', linestyle='--', label='
    plt.axvline(x=58.5, color='blue', linestyle='--', label='Tier 2 Threshold (58.5)')
    plt.axvline(x=45, color='orange', linestyle='--', label='Tier 3 Threshold (45)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Distribution of properties by tier
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=cat1_properties, x='PriorityTier', order=["Tier 1", "Tier 2", "Tier 3", "Tier 4"])
    plt.title('Number of Properties by Priority Tier', fontsize=16)
    plt.xlabel('Priority Tier', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Scatter plot of building size vs score
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=cat1_properties, x='BldgSqFt', y='DesirabilityScore', hue='PriorityTier', 
                    palette=['green', 'blue', 'orange', 'red'], alpha=0.7)
    plt.title('Building Size vs Desirability Score', fontsize=16)
    plt.xlabel('Building Square Footage', fontsize=12)
    plt.ylabel('Desirability Score', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Boxplot of scores by ZIP code
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=cat1_properties, x='SiteZIP', y='DesirabilityScore')
    plt.title('Desirability Score Distribution by ZIP Code', fontsize=16)
    plt.xlabel('ZIP Code', fontsize=12)
    plt.ylabel('Desirability Score', fontsize=12)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 11. Save Results
# 
# Save the filtered and scored properties to a CSV file for further analysis.

# %%
# Save the results to a CSV file
if len(cat1_properties) > 0:
    # Create output directory if it doesn't exist
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = f"{output_dir}/category1_scored_properties.csv"
    cat1_properties.to_csv(output_file, index=False)
    print(f"Saved {len(cat1_properties)} scored properties to {output_file}")

# %% [markdown]
# ## 12. Advanced Analysis - Individual Property Assessment
# 
# Select a single property to analyze in detail.

# %%
# Function to display detailed property information
def analyze_property(properties, property_index=0):
    """
    Display detailed information about a specific property
    
    Args:
        properties (pandas.DataFrame): Property data
        property_index (int): Index of the property to analyze
    """
    if len(properties) <= property_index:
        print(f"ERROR: Property index {property_index} is out of range. Only {len(properties)} properties available.")
        return
    
    # Get the property
    property = properties.iloc[property_index]
    
    # Display basic information
    print(f"PROPERTY ANALYSIS\n{'='*50}")
    print(f"Address: {property.get('SiteAddr', 'N/A')}, {property.get('SiteCity', 'N/A')}, {property.get('SiteState', 'N/A')} {property.get('SiteZIP', 'N/A')}")
    print(f"Property Type: {property.get('LandUseDsc', 'N/A')}")
    print(f"Market Value: ${property.get('MktTtlVal', 0):,.2f}")
    print(f"Year Built: {int(property.get('YrBlt', 0)) if pd.notna(property.get('YrBlt', 0)) else 'N/A'}")
    print(f"Building Size: {property.get('BldgSqFt', 0):,.0f} sq ft")
    print(f"Lot Size: {property.get('LotSqFt', 0):,.0f} sq ft ({property.get('Acres', 0):.2f} acres)")
    print(f"Bedrooms: {property.get('BedCt', 0)}")
    print(f"Bathrooms: {property.get('BathTtlCt', 0)} (Full: {property.get('BathFullCt', 0)}, 3/4: {property.get('Bath3QtrCt', 0)}, Half: {property.get('BathHalfCt', 0)})")
    print(f"Basement: {property.get('BsmtFinSqFt', 0):,.0f} sq ft finished, {property.get('BsmtUnFinSqFt', 0):,.0f} sq ft unfinished")
    print(f"Garage: {property.get('GarageDsc', 'N/A')}")
    print(f"Condition: {property.get('Condition', 'N/A')}")
    
    # Display scores if available
    if 'DesirabilityScore' in property:
        print(f"\nDESIRABILITY SCORES\n{'='*50}")
        print(f"Total Desirability Score: {property.get('DesirabilityScore', 0):.1f}")
        print(f"Priority Tier: {property.get('PriorityTier', 'N/A')}")
        
        # Display individual scores
        score_columns = [
            ('PropertyTypeScore', 'Property Type'),
            ('BathroomDistributionScore', 'Bathroom Distribution'),
            ('BuildingSizeScore', 'Building Size'),
            ('BasementSpaceScore', 'Basement Space'),
            ('StoriesCountScore', 'Stories Count'),
            ('ZipCodeValueScore', 'ZIP Code Value'),
            ('LotSizeScore', 'Lot Size'),
            ('ConditionScore', 'Condition'),
            ('YearBuiltScore', 'Year Built')
        ]
        
        # Get maximum score name length for formatting
        max_length = max(len(name) for _, name in score_columns)
        
        for col, name in score_columns:
            if col in property:
                print(f"{name + ':': <{max_length + 2}} {property.get(col, 0)}")
    
    # Display owner information if available
    if 'OwnerNmFirst' in property or 'OwnerNm' in property:
        print(f"\nOWNER INFORMATION\n{'='*50}")
        print(f"Owner: {property.get('OwnerNm', 'N/A')}")
        print(f"Owner Address: {property.get('OwnerAddr', 'N/A')}, {property.get('OwnerCityNm', 'N/A')}, {property.get('OwnerState', 'N/A')} {property.get('OwnerZIP', 'N/A')}")
        print(f"Owner Occupied: {'Yes' if property.get('OwnerOccupiedInd', False) else 'No'}")
    
    # Display tax information if available
    if 'TaxTtl1' in property:
        print(f"\nTAX INFORMATION\n{'='*50}")
        print(f"Current Tax (2024): ${property.get('TaxTtl1', 0):,.2f}")
        print(f"Previous Tax (2023): ${property.get('TaxTtl2', 0):,.2f}")
        print(f"2022 Tax: ${property.get('TaxTtl3', 0):,.2f}")
        
        # Calculate tax increase
        if pd.notna(property.get('TaxTtl1', None)) and pd.notna(property.get('TaxTtl2', None)) and property.get('TaxTtl2', 0) > 0:
            tax_increase = (property.get('TaxTtl1', 0) - property.get('TaxTtl2', 0)) / property.get('TaxTtl2', 0) * 100
            print(f"Year-over-Year Tax Increase: {tax_increase:.1f}%")
    
    return property

# %%
# Analyze the top-scoring property
if len(cat1_properties) > 0:
    # Sort by desirability score
    top_properties = cat1_properties.sort_values('DesirabilityScore', ascending=False)
    
    # Analyze the top property
    top_property = analyze_property(top_properties, property_index=0)

# %% [markdown]