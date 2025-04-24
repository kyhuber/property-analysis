#!/usr/bin/env python
# coding: utf-8

# # Property Analysis for Rent Ready Home Purchase
# 
# This notebook analyzes property data to identify properties that fit defined criteria and ranks them based on desirability and seller likelihood.
# 
# ## 1. Setup and Imports

# In[1]:


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

# In[2]:


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

# In[3]:


# Set path to your data file
file_path = "../data/Category_1.csv"

# Load and clean data
properties = load_data(file_path)


# ## 4. Examine the Data
# 
# Let's examine the data to understand what we're working with.

# In[4]:


# Display a few sample rows
properties.head()


# In[5]:


# Check data types and missing values
properties.info()


# In[6]:


# Look at statistics for numeric columns
properties.describe()


# In[7]:


# Check which ZIP codes are present
if 'SiteZIP' in properties.columns:
    print("ZIP code counts:")
    print(properties['SiteZIP'].value_counts())


# In[8]:


# Check distribution of property types
if 'LandUseDsc' in properties.columns:
    print("Land use types:")
    print(properties['LandUseDsc'].value_counts().head(10))


# ## 5. Filter Functions
# 
# Define functions to filter properties based on our criteria.

# In[9]:


def filter_category1(properties):
    """
    Apply Category 1 (Rent-Ready Properties) filters
    """
    print("Filtering for Category 1 (Rent-Ready Properties) with criteria:")
    print("  - At least 2 bathrooms")
    print("  - At least 3 bedrooms")
    print("  - In ZIP codes: 98106, 98116, 98126, 98136")
    print("  - Market value <= $850,000")
    print("  - Property type: Single Family or Duplex")

    # Normalize key fields
    if 'SiteZIP' in properties.columns:
        properties['SiteZIP'] = properties['SiteZIP'].astype(str).str.strip()
    
    if 'LandUseDsc' in properties.columns:
        properties['LandUseDsc'] = properties['LandUseDsc'].astype(str).str.strip()

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

    cat1_properties = properties[category1_filter].copy()

    print(f"Found {len(cat1_properties)} properties matching Category 1 criteria out of {len(properties)} total properties")
    
    if len(cat1_properties) > 0:
        print("\nBreakdown by ZIP code:")
        print(cat1_properties['SiteZIP'].value_counts())

    return cat1_properties

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

# In[10]:


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


# ## 7. Exploratory Data Analysis
# 
# Let's visualize some aspects of the filtered properties to better understand them.

# In[11]:


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


# ## 8. Scoring Functions - Desirability - Category 1
# 
# Now let's define functions to scorecat1 each property on desirability factors.

# In[12]:


#========== DESIRABILITY SCORE FUNCTIONS - CATEGORY 1==========


def calculate_property_type_scorecat1(row):
    """
    Calculate property type scorecat1 (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat1 between 0-15
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




def calculate_bathroom_distribution_scorecat1(row):
    """
    Calculate bathroom distribution scorecat1 (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat1 between 0-15
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




def calculate_building_size_scorecat1(row):
    """
    Calculate building size scorecat (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat1 between 0-10
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


def calculate_basement_space_scorecat1(row):
    """
    Calculate basement space score for Category 1 (10 points max)
    For conversion-ready properties, unfinished basement space is preferred
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    bsmt_fin = row['BsmtFinSqFt'] if pd.notna(row['BsmtFinSqFt']) else 0
    bsmt_unfin = row['BsmtUnFinSqFt'] if pd.notna(row['BsmtUnFinSqFt']) else 0
    
    # For conversion-ready properties, unfinished space is preferred
    if bsmt_fin > 800:
        return 10  # Large unfinished basement (ideal for conversion)
    elif bsmt_fin >= 500:
        return 8   # Good unfinished space for conversion
    elif bsmt_fin >= 300:
        return 6   # Mix of finished/unfinished space
    elif bsmt_fin > 200:
        return 4   # Already finished (less flexible for conversion)
    elif (bsmt_fin + bsmt_unfin) > 0:
        return 2   # Some basement is better than none
    else:
        return 0   # No basement


def calculate_stories_count_scorecat1(row):
    """
    Calculate stories count scorecat1 (5 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat1 between 0-5
    """
    bsmt_fin = row['BsmtFinSqFt'] if pd.notna(row['BsmtFinSqFt']) else 0
    bsmt_unfin = row['BsmtUnFinSqFt'] if pd.notna(row['BsmtUnFinSqFt']) else 0
    
    if pd.notna(row['StoriesCt']) and row['StoriesCt'] >= 2:
        return 5
    elif pd.notna(row['StoriesCt']) and row['StoriesCt'] == 1 and (bsmt_fin > 0 or bsmt_unfin > 0):
        return 3
    else:
        return 1




def calculate_zip_code_value_scorecat1(row):
    """
    Calculate ZIP code value scorecat1 (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat1 between 0-15
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




def calculate_lot_size_scorecat1(row):
    """
    Calculate lot size scorecat1 (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat1 between 0-10
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




def calculate_condition_scorecat1(row):
    """
    Calculate condition scorecat1 (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat1 between 0-10
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




def calculate_year_built_scorecat1(row):
    """
    Calculate year built scorecat1 (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: ScoreCat1 between 0-10
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

def calculate_zoning_scorecat1(row):
    """
    Calculate zoning score for Category 1 (Rent-Ready Properties) (10 points max)
    Evaluates zoning compatibility with existing or potential rental units
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['ZoneCd']):
        zone_code = str(row['ZoneCd']).strip()
        
        # Duplex/multi-family zones score highest
        if any(code in zone_code for code in ['LR2', 'LR3', 'RD', 'MDR']):
            return 10  # Lowrise multi-family or residential duplex zones - ideal for rentals
        
        # Zones that explicitly allow ADUs or additional units
        elif 'RSL' in zone_code or 'DADU' in zone_code:
            return 9  # Residential Small Lot or Detached Accessory Dwelling Unit zones
            
        # Higher density residential zones
        elif any(code in zone_code for code in ['R-18', 'R-24', 'NR3']):
            return 8  # Higher density residential zones with rental potential
            
        # Medium density residential
        elif any(code in zone_code for code in ['NR2', 'R-12']):
            return 6  # Medium density residential
            
        # Lower density residential with potential for conversion
        elif any(code in zone_code for code in ['R-6', 'R-8']):
            return 4  # Lower density residential, but still with some potential
            
        # Other residential zones not clearly categorized
        elif 'R-' in zone_code:
            return 3  # Other residential zones with limited rental potential
            
        # Non-residential or industrial zones
        elif 'UI' in zone_code or 'IC' in zone_code:
            return 0  # Urban Industrial or Industrial Commercial - not suitable for residential
            
        else:
            return 2  # Unknown zoning
    
    return 0  # No zoning information available


# In[13]:


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
        'YrBlt': 1985 ,
        'ZoneCd' : 'Residential Small Lot'
    })
    
    # Test each scoring function
    print("Testing desirability scoring functions on sample property...")
    print(f"Property Type ScoreCat1: {calculate_property_type_scorecat1(sample_property)}")
    print(f"Bathroom Distribution ScoreCat1: {calculate_bathroom_distribution_scorecat1(sample_property)}")
    print(f"Building Size ScoreCat1: {calculate_building_size_scorecat1(sample_property)}")
    print(f"Basement Space ScoreCat1: {calculate_basement_space_scorecat1(sample_property)}")
    print(f"Stories Count ScoreCat1: {calculate_stories_count_scorecat1(sample_property)}")
    print(f"ZIP Code Value ScoreCat1: {calculate_zip_code_value_scorecat1(sample_property)}")
    print(f"Lot Size ScoreCat1: {calculate_lot_size_scorecat1(sample_property)}")
    print(f"Condition ScoreCat1: {calculate_condition_scorecat1(sample_property)}")
    print(f"Year Built ScoreCat1: {calculate_year_built_scorecat1(sample_property)}")
    
    # Calculate total desirability scorecat1
    total_scorecat1 = (
        calculate_property_type_scorecat1(sample_property) +
        calculate_bathroom_distribution_scorecat1(sample_property) +
        calculate_building_size_scorecat1(sample_property) +
        calculate_basement_space_scorecat1(sample_property) +
        calculate_stories_count_scorecat1(sample_property) +
        calculate_zip_code_value_scorecat1(sample_property) +
        calculate_lot_size_scorecat1(sample_property) +
        calculate_condition_scorecat1(sample_property) +
        calculate_year_built_scorecat1(sample_property) +
        calculate_zoning_scorecat1(sample_property)
    )
    
    print(f"Total Desirability ScoreCat1: {total_scorecat1}")




# Run the test to see if our scoring functions work correctly
test_desirability_scoring_functions()


# ## 9. Calculate Cat1 Seller Likelihood
# 
# Determine seller likelihood for all filtered properties.

# In[14]:


def calculate_seller_likelihood_score(properties_df):
    """
    Calculate the likelihood of sellers being willing to sell their properties,
    regardless of the property's characteristics for any specific buyer category.
    
    Parameters:
    -----------
    properties_df : pandas DataFrame
        DataFrame containing property information
    
    Returns:
    --------
    pandas DataFrame
        Original DataFrame with added seller likelihood score and factors
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = properties_df.copy()
    
    # Initialize the seller likelihood score column and factors description
    df['SellerLikelihoodScore'] = 0
    df['SellerLikelihoodFactors'] = ''
    
    # ===== OWNERSHIP FACTORS =====
    
    # Length of ownership - properties owned longer may be more likely to sell
    if 'DocRcrdgDt_County' in df.columns:
        df['OwnershipLength'] = pd.Timestamp.now().year - pd.to_datetime(df['DocRcrdgDt_County']).dt.year
        
        # Apply scores based on ownership length
        df.loc[df['OwnershipLength'] >= 15, 'SellerLikelihoodScore'] += 20
        df.loc[(df['OwnershipLength'] >= 7) & (df['OwnershipLength'] < 15), 'SellerLikelihoodScore'] += 15
        df.loc[(df['OwnershipLength'] >= 3) & (df['OwnershipLength'] < 7), 'SellerLikelihoodScore'] += 7
        
        # Add factor descriptions
        df.loc[df['OwnershipLength'] >= 15, 'SellerLikelihoodFactors'] += 'Very long-term owner; '
        df.loc[(df['OwnershipLength'] >= 7) & (df['OwnershipLength'] < 15), 'SellerLikelihoodFactors'] += 'Long-term owner; '
    
    # Non-owner occupied properties may be more likely to sell
    if 'OwnerOccupiedInd' in df.columns:
        df.loc[df['OwnerOccupiedInd'] == False, 'SellerLikelihoodScore'] += 20
        df.loc[df['OwnerOccupiedInd'] == False, 'SellerLikelihoodFactors'] += 'Investment property; '
    
    # Corporate ownership might indicate investment property
    if 'OwnerCorporateInd' in df.columns:
        df.loc[df['OwnerCorporateInd'] == True, 'SellerLikelihoodScore'] += 15
        df.loc[df['OwnerCorporateInd'] == True, 'SellerLikelihoodFactors'] += 'Corporate owner; '
    
    # Different mailing address may indicate non-local owner
    if all(col in df.columns for col in ['OwnerAddr', 'SiteAddr']):
        df['DifferentMailingAddr'] = df['OwnerAddr'] != df['SiteAddr']
        df.loc[df['DifferentMailingAddr'], 'SellerLikelihoodScore'] += 15
        df.loc[df['DifferentMailingAddr'], 'SellerLikelihoodFactors'] += 'Non-local owner; '
    
    # ===== FINANCIAL FACTORS =====
    
    # Tax assessment increases - rapidly increasing property taxes may motivate selling
    if all(col in df.columns for col in ['TaxTtl1', 'TaxTtl2', 'TaxTtl3']):
        # Convert tax columns to numeric if needed
        for col in ['TaxTtl1', 'TaxTtl2', 'TaxTtl3']:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(r'[\$,]', '', regex=True), errors='coerce')
        
        # Calculate year-over-year tax increases as percentage
        df['TaxChange_Recent'] = ((df['TaxTtl1'] - df['TaxTtl2']) / df['TaxTtl2'] * 100).fillna(0)
        df['TaxChange_Previous'] = ((df['TaxTtl2'] - df['TaxTtl3']) / df['TaxTtl3'] * 100).fillna(0)
        
        # Score based on tax increases
        df.loc[df['TaxChange_Recent'] >= 15, 'SellerLikelihoodScore'] += 15
        df.loc[(df['TaxChange_Recent'] >= 10) & (df['TaxChange_Recent'] < 15), 'SellerLikelihoodScore'] += 10
        df.loc[(df['TaxChange_Recent'] >= 5) & (df['TaxChange_Recent'] < 10), 'SellerLikelihoodScore'] += 5
        
        # Consistent tax increases over multiple years
        df.loc[(df['TaxChange_Recent'] >= 8) & (df['TaxChange_Previous'] >= 8), 'SellerLikelihoodScore'] += 8
        
        # Add factor descriptions
        df.loc[df['TaxChange_Recent'] >= 15, 'SellerLikelihoodFactors'] += 'Extreme tax increase; '
        df.loc[(df['TaxChange_Recent'] >= 10) & (df['TaxChange_Recent'] < 15), 'SellerLikelihoodFactors'] += 'Significant tax increase; '
        df.loc[(df['TaxChange_Recent'] >= 5) & (df['TaxChange_Previous'] >= 5), 'SellerLikelihoodFactors'] += 'Sustained tax increases; '
    
    # Value-to-tax ratio may indicate pressure to sell
    if all(col in df.columns for col in ['MktTtlVal', 'TaxTtl1']):
        df['TaxToValueRatio'] = (df['TaxTtl1'] / df['MktTtlVal'] * 100000).fillna(0)
        
        df.loc[df['TaxToValueRatio'] > 1.5, 'SellerLikelihoodScore'] += 10
        df.loc[df['TaxToValueRatio'] > 1.5, 'SellerLikelihoodFactors'] += 'High tax burden relative to value; '
    
    # Recent market value changes may impact selling motivation
    if 'MktTtlVal' in df.columns and 'AssdTtlVal' in df.columns:
        # Rapidly appreciating property might be attractive to sell
        df['ValueAssessmentRatio'] = (df['MktTtlVal'] / df['AssdTtlVal']).fillna(1)
        
        df.loc[df['ValueAssessmentRatio'] > 1.25, 'SellerLikelihoodScore'] += 10
        df.loc[df['ValueAssessmentRatio'] > 1.25, 'SellerLikelihoodFactors'] += 'Significant recent appreciation; '
    
    # Length of ownership combined with age may indicate life transition
    if 'OwnershipLength' in df.columns:
        # Very long ownership might indicate aging owners considering downsizing
        df.loc[df['OwnershipLength'] >= 25, 'SellerLikelihoodScore'] += 8
        df.loc[df['OwnershipLength'] >= 25, 'SellerLikelihoodFactors'] += 'Potential life transition; '
    
    # ===== SCORE NORMALIZATION =====
    
    # Calculate the maximum possible score
    max_possible_score = 121  # Sum of all maximum points from factors above
    
    # Normalize to 0-100 scale
    df['SellerLikelihoodScore'] = (df['SellerLikelihoodScore'] / max_possible_score) * 100
    
    # Ensure scores stay within 0-100 range
    df['SellerLikelihoodScore'] = df['SellerLikelihoodScore'].clip(0, 100)
    
    # Create likelihood categories
    df['SellerLikelihoodCategory'] = pd.cut(
        df['SellerLikelihoodScore'], 
        bins=[0, 25, 50, 75, 100], 
        labels=['Low', 'Moderate', 'High', 'Very High']
    )
    
    # Clean up factors string (remove trailing semicolon and space)
    df['SellerLikelihoodFactors'] = df['SellerLikelihoodFactors'].str.rstrip('; ')
    
    return df


# ## 10. Calculate ScoreCat1s for All Properties
# 
# Apply the scoring functions to all filtered properties.

# In[15]:


def calculate_all_scorecat1s(properties):
    """
    Calculate all individual scorecat1s for properties
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Property data with scorecat1s
    """
    # Calculate desirability scorecat1s
    print("Calculating desirability scorecat1s...")
    properties['PropertyTypeScoreCat1'] = properties.apply(calculate_property_type_scorecat1, axis=1)
    properties['BathroomDistributionScoreCat1'] = properties.apply(calculate_bathroom_distribution_scorecat1, axis=1)
    properties['BuildingSizeScoreCat1'] = properties.apply(calculate_building_size_scorecat1, axis=1)
    properties['BasementSpaceScoreCat1'] = properties.apply(calculate_basement_space_scorecat1, axis=1)
    properties['StoriesCountScoreCat1'] = properties.apply(calculate_stories_count_scorecat1, axis=1)
    properties['ZipCodeValueScoreCat1'] = properties.apply(calculate_zip_code_value_scorecat1, axis=1)
    properties['LotSizeScoreCat1'] = properties.apply(calculate_lot_size_scorecat1, axis=1)
    properties['ConditionScoreCat1'] = properties.apply(calculate_condition_scorecat1, axis=1)
    properties['YearBuiltScoreCat1'] = properties.apply(calculate_year_built_scorecat1, axis=1)
    properties['ZoningScoreCat1'] = properties.apply(calculate_zoning_scorecat1, axis=1)
    
    return properties




def calculate_combined_scorecat1s(properties):
    """
    Calculate total and combined scorecat1s, and assign priority tiers
    
    Args:
        properties (pandas.DataFrame): Property data with individual scorecat1s
        
    Returns:
        pandas.DataFrame: Property data with total scorecat1s and priority tiers
    """
    # Calculate desirability scorecat1
    desirability_columns = [
        'PropertyTypeScoreCat1', 'BathroomDistributionScoreCat1', 'BuildingSizeScoreCat1',
        'BasementSpaceScoreCat1', 'StoriesCountScoreCat1', 'ZipCodeValueScoreCat1',
        'LotSizeScoreCat1', 'ConditionScoreCat1', 'YearBuiltScoreCat1', 'ZoningScoreCat1'
    ]
    
    properties['DesirabilityScoreCat1'] = properties[desirability_columns].sum(axis=1)
    
    # Calculate seller likelihood score
    properties = calculate_seller_likelihood_score(properties)
    
    # Calculate final combined score (weighted average)
    properties['FinalScoreCat1'] = (
        properties['DesirabilityScoreCat1'] * 0.7 + 
        properties['SellerLikelihoodScore'] * 0.3
    )
    
    # Assign priority tiers based on the combined final score
    def assign_tier(scorecat1):
        # Calculate max possible score
        max_desirability = 100  # 15+15+10+10+5+15+10+10+10+10 (includes zoning)
        max_seller = 110  # Total possible seller likelihood points
        max_combined = max_desirability * 0.7 + max_seller * 0.3
        
        if scorecat1 >= max_combined * 0.8:  # 80% or higher
            return "Tier 1"
        elif scorecat1 >= max_combined * 0.65:  # 65-80%
            return "Tier 2"
        elif scorecat1 >= max_combined * 0.5:  # 50-65%
            return "Tier 3"
        else:  # Below 50%
            return "Tier 4"
    
    properties['PriorityTier'] = properties['FinalScoreCat1'].apply(assign_tier)
    
    # Calculate max possible scores for reference
    max_desirability = 100
    max_seller = 110
    max_combined = max_desirability * 0.7 + max_seller * 0.3
    
    print("ScoreCat1 ranges for each tier:")
    print(f"  Tier 1: {max_combined * 0.8:.1f}-{max_combined:.1f}")
    print(f"  Tier 2: {max_combined * 0.65:.1f}-{max_combined * 0.8:.1f}")
    print(f"  Tier 3: {max_combined * 0.5:.1f}-{max_combined * 0.65:.1f}")
    print(f"  Tier 4: 0-{max_combined * 0.5:.1f}")
    
    return properties


# In[16]:


# Apply scoring functions to Category 1 properties
if len(cat1_properties) > 0:
    # Calculate scorecat1s
    cat1_properties = calculate_all_scorecat1s(cat1_properties)
    cat1_properties = calculate_combined_scorecat1s(cat1_properties)
    
    # Display scorecat1 summary
    print("\nScoreCat1 summary statistics:")
    print(cat1_properties['DesirabilityScoreCat1'].describe())
    
    # Count properties by tier
    print("\nProperties by tier:")
    tier_counts = cat1_properties['PriorityTier'].value_counts().sort_index()
    print(tier_counts)
    
    # Display top 5 properties
    display_columns = [
        'SiteAddr', 'SiteCity', 'SiteZIP', 'BathTtlCt', 'BedCt',
        'BldgSqFt', 'YrBlt', 'MktTtlVal', 'DesirabilityScoreCat1', 'PriorityTier', 'ZoneCd'
    ]
    
    # Make sure all requested columns exist
    display_columns = [col for col in display_columns if col in cat1_properties.columns]
    
    print("\nTop 5 properties by final score:")
    top_properties = cat1_properties.sort_values('FinalScoreCat1', ascending=False)
    display(top_properties[display_columns].head())


# ## 11.  Reorder Results

# In[ ]:


def reorder_columns(df, category):
    """
    Reorder columns based on the property category and recommended structure
    
    Parameters:
    df (DataFrame): The DataFrame containing property data and scores
    category (int): The category number (1, 2, or 3)
    
    Returns:
    DataFrame: A new DataFrame with reordered columns
    """
    # Define column groups
    scoring_cols = [
        'PriorityTier', 'DesirabilityScoreCat1', 'SellerLikelihoodScore', 'FinalScoreCat1'
    ]
    
    # Add factor score columns based on category
    if category == 1:  # Rent-Ready Properties
        factor_cols = [
            'PropertyTypeScoreCat1', 'BathroomDistributionScoreCat1', 'BuildingSizeScoreCat1',
            'BasementSpaceScoreCat1', 'StoriesCountScoreCat1', 'ZipCodeValueScoreCat1',
            'LotSizeScoreCat1', 'ConditionScoreCat1', 'YearBuiltScoreCat1', 'ZoningScoreCat1'
        ]
    
    # Basic property identification
    id_cols = [
        'SiteAddr', 'SiteCity', 'SiteState', 'SiteZIP', 'ParcelId', 'LandUseDsc'
    ]
    
    # Key property characteristics
    key_cols = [
        'MktTtlVal', 'BedCt', 'BathTtlCt', 'BldgSqFt', 'LotSqFt', 'Acres',
        'YrBlt', 'Condition'
    ]
    
    # Category-specific features
    if category == 1:  # Rent-Ready Properties
        category_cols = [
            'GarageDsc', 'BsmtFinSqFt', 'BsmtUnFinSqFt', 'StoriesCt', 'FireplaceCt'
        ]
    
    # Financial/tax information
    financial_cols = [
        'TaxTtl1', 'TaxYr1', 'AssdImprVal', 'AssdLandVal', 'AssdTtlVal'
    ]
    
    # Owner information
    owner_cols = [
        'OwnerNmFirstBoth', 'OwnerNmLast', 'OwnerOccupiedInd'
    ]
    
    # Combine all column groups in desired order
    ordered_cols = scoring_cols + factor_cols + id_cols + key_cols + category_cols + financial_cols + owner_cols
    
    # Filter to only include columns that exist in the DataFrame
    existing_cols = [col for col in ordered_cols if col in df.columns]
    
    # Add any remaining columns that weren't explicitly ordered
    remaining_cols = [col for col in df.columns if col not in existing_cols]
    final_ordered_cols = existing_cols + remaining_cols
    
    # Return DataFrame with reordered columns
    return df[final_ordered_cols]


# ## 12. Save Results
# 
# Save the filtered and scorecat1d properties to a CSV file for further analysis.

# In[18]:


# Save the results to a CSV file
if len(cat1_properties) > 0:
    # Create output directory if it doesn't exist
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = f"{output_dir}/category1_properties_rent_ready.csv"
    cat1_properties.to_csv(output_file, index=False)
    print(f"Saved {len(cat1_properties)} category 1 rent ready properties to {output_file}")


# In[ ]:




