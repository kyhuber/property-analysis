#!/usr/bin/env python
# coding: utf-8

# # Property Analysis for Home Purchase
# 
# This notebook analyzes property data to identify properties that fit your criteria and ranks them based on desirability and seller likelihood.
# 
# ## 1. Setup and Imports

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import re


# Enable interactive plots
get_ipython().run_line_magic('matplotlib', 'inline')


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


# ## 2. Load The Data
# 
# Load the property data, with an option to test on a small subset first.

# In[20]:


def load_data(file_path):
    """
    Load and clean property data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Cleaned property data
    """
    print(f"Loading data from {file_path}...")
    
    # Use dtype parameter to specify data types upfront for more efficient loading
    # Also use low_memory=False to avoid mixed type inference warnings
    dtypes = {
        'SiteZIP': str,
        'LandUseDsc': str
    }
    
    properties = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
    print(f"Original data shape: {properties.shape}")
    
    # Convert date columns - with a specific format for efficiency
    if 'DocRcrdgDt_County' in properties.columns:
        properties['DocRcrdgDt_County'] = pd.to_datetime(
            properties['DocRcrdgDt_County'], errors='coerce', format='%Y-%m-%d'
        )
    
    # Use vectorized operations instead of astype+str+replace chain
    currency_columns = [
        'MktTtlVal', 'SaleAmt_County', 'TaxTtl1', 'TaxTtl2', 'TaxTtl3'
    ]
    for col in currency_columns:
        if col in properties.columns:
            # This is more efficient than the chain of operations
            properties[col] = pd.to_numeric(
                properties[col].astype(str).str.replace(r'[\$,]', '', regex=True),
                errors='coerce'
            )
    
    # Handle other numeric columns in a batch when possible
    other_numeric_columns = [
        'BsmtFinSqFt', 'BsmtUnFinSqFt', 'BathHalfCt', 'Bath3QtrCt',
        'BathFullCt', 'BathTtlCt', 'BedCt', 'BldgSqFt', 'LotSqFt',
        'YrBlt', 'StoriesCt', 'Acres'
    ]
    
    # Check which columns exist first
    existing_num_cols = [col for col in other_numeric_columns if col in properties.columns]
    if existing_num_cols:
        properties[existing_num_cols] = properties[existing_num_cols].apply(
            pd.to_numeric, errors='coerce'
        )
    
    # Use boolean conversion directly where possible
    boolean_columns = [
        'OwnerOccupiedInd', 'BareLandInd', 'InvestmentProp', 
        'BankOwnedInd', 'OwnerCorporateInd'
    ]
    for col in boolean_columns:
        if col in properties.columns:
            properties[col] = properties[col].map({
                'TRUE': True, 'FALSE': False, True: True, False: False
            })
    
    print(f"Cleaned data shape: {properties.shape}")
    print(f"Sample market values: {properties['MktTtlVal'].dropna().head().tolist()}")
    
    return properties


# ### 3. Data Loading Function
# 
# Define a function to load and clean the property data CSV.

# In[21]:


# Set path to your data file
file_path = "../data/Category_2.csv"

# Load and clean data
properties = load_data(file_path)


# ## 4. Examine the Data
# 
# Let's examine the data to understand what we're working with.

# In[22]:


# Display a few sample rows
properties.head()


# In[23]:


# Check data types and missing values
properties.info()


# In[24]:


# Look at statistics for numeric columns
properties.describe()


# In[25]:


# Check which ZIP codes are present
if 'SiteZIP' in properties.columns:
    print("ZIP code counts:")
    print(properties['SiteZIP'].value_counts())


# In[26]:


# Check distribution of property types
if 'LandUseDsc' in properties.columns:
    print("Land use types:")
    print(properties['LandUseDsc'].value_counts().head(10))


# ## 5. Filter Functions
# 
# Define functions to filter properties based on our criteria.

# In[27]:


def filter_category2(properties):
    """
    Apply Category 2 (Conversion-Ready Properties) filters
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Filtered property data
    """
    print("Filtering for Category 2 (Conversion-Ready Properties) with criteria:")
    print("  - At least 1.5 bathrooms")
    print("  - At least 3 bedrooms")
    print("  - Building size >= 1600 sq ft")
    print("  - Lot size > 5000 sq ft")
    print("  - In ZIP codes: 98106, 98116, 98126, 98136")
    print("  - Market value between $650,000-$800,000")
    print("  - Property type: Single Family")

    # Normalize key fields
    if 'SiteZIP' in properties.columns:
        properties['SiteZIP'] = properties['SiteZIP'].astype(str).str.strip()
    
    if 'LandUseDsc' in properties.columns:
        properties['LandUseDsc'] = properties['LandUseDsc'].astype(str).str.strip()

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

    cat2_properties = properties[category2_filter].copy()

    print(f"Found {len(cat2_properties)} properties matching Category 2 criteria out of {len(properties)} total properties")
    
    if len(cat2_properties) > 0:
        print("\nBreakdown by ZIP code:")
        print(cat2_properties['SiteZIP'].value_counts())

    return cat2_properties

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


# ## 6. Apply Filters
# 
# Let's apply the filters and see what properties match our criteria.

# In[28]:


# Apply Category 2 filters
cat2_properties = filter_category2(properties)


# Display the first few properties that match
if len(cat2_properties) > 0:
    # Select a subset of columns to display
    display_columns = [
        'SiteAddr', 'SiteCity', 'SiteZIP', 'BathTtlCt', 'BedCt',
        'BldgSqFt', 'LotSqFt', 'YrBlt', 'MktTtlVal', 'LandUseDsc'
    ]
    
    # Make sure all requested columns exist
    display_columns = [col for col in display_columns if col in cat2_properties.columns]
    
    print("\nSample of matching Category 2 properties:")
    display(cat2_properties[display_columns].head())
else:
    print("\nNo properties match Category 2 criteria.")


# Calculate neighborhood averages
if len(cat2_properties) > 0:
    zip_averages = calculate_zip_averages(cat2_properties)
    
    # Merge with the properties dataframe
    cat2_properties = cat2_properties.merge(zip_averages, on='SiteZIP', how='left')


# ## 7. Exploratory Data Analysis
# 
# Let's visualize some aspects of the filtered properties to better understand them.

# In[29]:


# Only run if we have matching properties
if len(cat2_properties) > 0:
    # Distribution of properties by ZIP code
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=cat2_properties, x='SiteZIP')
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
    sns.histplot(cat2_properties['MktTtlVal'], bins=20, kde=True)
    plt.title('Distribution of Property Values', fontsize=16)
    plt.xlabel('Market Total Value ($)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Building Size vs Market Value
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=cat2_properties, x='BldgSqFt', y='MktTtlVal', hue='SiteZIP', alpha=0.7)
    plt.title('Building Size vs Market Value by ZIP Code', fontsize=16)
    plt.xlabel('Building Square Footage', fontsize=12)
    plt.ylabel('Market Value ($)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Boxplot of property values by ZIP code
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=cat2_properties, x='SiteZIP', y='MktTtlVal')
    plt.title('Property Value Distribution by ZIP Code', fontsize=16)
    plt.xlabel('ZIP Code', fontsize=12)
    plt.ylabel('Market Value ($)', fontsize=12)
    plt.tight_layout()
    plt.show()


# ## 8. Scoring Functions - Desirability - Category 2
# 
# Now let's define functions to scorecat2 each property on desirability factors.

# In[30]:


#========== DESIRABILITY SCORE FUNCTIONS - CATEGORY 2==========


def calculate_property_type_scorecat2(row):
    """
    Calculate property type scorecat2 (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-15
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




def calculate_bathroom_distribution_scorecat2(row):
    """
    Calculate bathroom distribution scorecat2 (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-15
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




def calculate_building_size_scorecat2(row):
    """
    Calculate building size scorecat (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-10
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




def calculate_basement_space_scorecat2(row):
    """
    Calculate basement space scorecat2 (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-10
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




def calculate_stories_count_scorecat2(row):
    """
    Calculate stories count scorecat2 (5 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-5
    """
    bsmt_fin = row['BsmtFinSqFt'] if pd.notna(row['BsmtFinSqFt']) else 0
    bsmt_unfin = row['BsmtUnFinSqFt'] if pd.notna(row['BsmtUnFinSqFt']) else 0
    
    if pd.notna(row['StoriesCt']) and row['StoriesCt'] >= 2:
        return 5
    elif pd.notna(row['StoriesCt']) and row['StoriesCt'] == 1 and (bsmt_fin > 0 or bsmt_unfin > 0):
        return 3
    else:
        return 1




def calculate_zip_code_value_scorecat2(row):
    """
    Calculate ZIP code value scorecat2 (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-15
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




def calculate_lot_size_scorecat2(row):
    """
    Calculate lot size scorecat2 (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-10
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




def calculate_condition_scorecat2(row):
    """
    Calculate condition scorecat2 (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-10
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




def calculate_year_built_scorecat2(row):
    """
    Calculate year built scorecat2 (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat2 between 0-10
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


# In[31]:


# Test these functions on a sample property to see if they work as expected
def test_desirability_scoring_functions():
    # Create a sample property for testing
    sample_property = pd.Series({
        'LandUseDsc': 'Single Family(Res Use/Zone)',
        'BsmtFinSqFt': 700,
        'BsmtUnFinSqFt': 500,
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
    print(f"Property Type ScoreCat2: {calculate_property_type_scorecat2(sample_property)}")
    print(f"Bathroom Distribution ScoreCat2: {calculate_bathroom_distribution_scorecat2(sample_property)}")
    print(f"Building Size ScoreCat2: {calculate_building_size_scorecat2(sample_property)}")
    print(f"Basement Space ScoreCat2: {calculate_basement_space_scorecat2(sample_property)}")
    print(f"Stories Count ScoreCat2: {calculate_stories_count_scorecat2(sample_property)}")
    print(f"ZIP Code Value ScoreCat2: {calculate_zip_code_value_scorecat2(sample_property)}")
    print(f"Lot Size ScoreCat2: {calculate_lot_size_scorecat2(sample_property)}")
    print(f"Condition ScoreCat2: {calculate_condition_scorecat2(sample_property)}")
    print(f"Year Built ScoreCat2: {calculate_year_built_scorecat2(sample_property)}")
    
    # Calculate total desirability scorecat2
    total_scorecat2 = (
        calculate_property_type_scorecat2(sample_property) +
        calculate_bathroom_distribution_scorecat2(sample_property) +
        calculate_building_size_scorecat2(sample_property) +
        calculate_basement_space_scorecat2(sample_property) +
        calculate_stories_count_scorecat2(sample_property) +
        calculate_zip_code_value_scorecat2(sample_property) +
        calculate_lot_size_scorecat2(sample_property) +
        calculate_condition_scorecat2(sample_property) +
        calculate_year_built_scorecat2(sample_property)
    )
    
    print(f"Total Desirability ScoreCat2: {total_scorecat2}")




# Run the test to see if our scoring functions work correctly
test_desirability_scoring_functions()


# ## 9. Calculate ScoreCat2s for All Properties
# 
# Apply the scoring functions to all filtered properties.

# In[32]:


def calculate_all_scorecat2s(properties):
    """
    Calculate all individual scorecat2s for properties
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Property data with scorecat2s
    """
    # Calculate desirability scorecat2s
    print("Calculating desirability scorecat2s...")
    properties['PropertyTypeScoreCat2'] = properties.apply(calculate_property_type_scorecat2, axis=1)
    properties['BathroomDistributionScoreCat2'] = properties.apply(calculate_bathroom_distribution_scorecat2, axis=1)
    properties['BuildingSizeScoreCat2'] = properties.apply(calculate_building_size_scorecat2, axis=1)
    properties['BasementSpaceScoreCat2'] = properties.apply(calculate_basement_space_scorecat2, axis=1)
    properties['StoriesCountScoreCat2'] = properties.apply(calculate_stories_count_scorecat2, axis=1)
    properties['ZipCodeValueScoreCat2'] = properties.apply(calculate_zip_code_value_scorecat2, axis=1)
    properties['LotSizeScoreCat2'] = properties.apply(calculate_lot_size_scorecat2, axis=1)
    properties['ConditionScoreCat2'] = properties.apply(calculate_condition_scorecat2, axis=1)
    properties['YearBuiltScoreCat2'] = properties.apply(calculate_year_built_scorecat2, axis=1)
    
    return properties




def calculate_combined_scorecat2s(properties):
    """
    Calculate total and combined scorecat2s, and assign priority tiers
    
    Args:
        properties (pandas.DataFrame): Property data with individual scorecat2s
        
    Returns:
        pandas.DataFrame: Property data with total scorecat2s and priority tiers
    """
    # Calculate desirability scorecat2
    desirability_columns = [
        'PropertyTypeScoreCat2', 'BathroomDistributionScoreCat2', 'BuildingSizeScoreCat2',
        'BasementSpaceScoreCat2', 'StoriesCountScoreCat2', 'ZipCodeValueScoreCat2',
        'LotSizeScoreCat2', 'ConditionScoreCat2', 'YearBuiltScoreCat2'
    ]
    
    properties['DesirabilityScoreCat2'] = properties[desirability_columns].sum(axis=1)
    
    # Calculate seller likelihood scorecat2 if we decide to implement those functions
    
    # For now, just use DesirabilityScoreCat2 for ranking
    properties['CombinedScoreCat2'] = properties['DesirabilityScoreCat2']
    
    # Assign priority tiers
    def assign_tier(scorecat2):
        max_possible = 90  # 15+15+10+10+5+15+10+10+10
        if scorecat2 >= max_possible * 0.8:  # 80% or higher
            return "Tier 1"
        elif scorecat2 >= max_possible * 0.65:  # 65-80%
            return "Tier 2"
        elif scorecat2 >= max_possible * 0.5:  # 50-65%
            return "Tier 3"
        else:  # Below 50%
            return "Tier 4"
    
    properties['PriorityTier'] = properties['CombinedScoreCat2'].apply(assign_tier)
    
    print("ScoreCat2 ranges for each tier:")
    print("  Tier 1: 72-90")
    print("  Tier 2: 58.5-71.9")
    print("  Tier 3: 45-58.4")
    print("  Tier 4: 0-44.9")
    
    return properties


# In[33]:


# Apply scoring functions to Category 2 properties
if len(cat2_properties) > 0:
    # Calculate scorecat2s
    cat2_properties = calculate_all_scorecat2s(cat2_properties)
    cat2_properties = calculate_combined_scorecat2s(cat2_properties)
    
    # Display scorecat2 summary
    print("\nScoreCat2 summary statistics:")
    print(cat2_properties['DesirabilityScoreCat2'].describe())
    
    # Count properties by tier
    print("\nProperties by tier:")
    tier_counts = cat2_properties['PriorityTier'].value_counts().sort_index()
    print(tier_counts)
    
    # Display top 5 properties
    display_columns = [
        'SiteAddr', 'SiteCity', 'SiteZIP', 'BathTtlCt', 'BedCt',
        'BldgSqFt', 'YrBlt', 'MktTtlVal', 'DesirabilityScoreCat2', 'PriorityTier'
    ]
    
    # Make sure all requested columns exist
    display_columns = [col for col in display_columns if col in cat2_properties.columns]
    
    print("\nTop 5 properties by scorecat2:")
    top_properties = cat2_properties.sort_values('DesirabilityScoreCat2', ascending=False)
    display(top_properties[display_columns].head())


# ## 11. Save Results
# 
# Save the filtered and scorecat2d properties to a CSV file for further analysis.

# In[34]:


# Save the results to a CSV file
if len(cat2_properties) > 0:
    # Create output directory if it doesn't exist
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = f"{output_dir}/category2_scorecat2_properties_all.csv"
    cat2_properties.to_csv(output_file, index=False)
    print(f"Saved {len(cat2_properties)} scorecat2d properties to {output_file}")


# In[ ]:




