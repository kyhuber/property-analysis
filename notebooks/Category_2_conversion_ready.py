#!/usr/bin/env python
# coding: utf-8

# # Property Analysis for Conversion Ready Home Purchase
# 
# This notebook analyzes property data to identify properties that fit defined criteria and ranks them based on desirability and seller likelihood.
# 
# ## 1. Setup and Imports

# In[21]:


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

# In[22]:


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
        'YrBlt', 'StoriesCt', 'Acres', 'ZoneCd', 'ZoneDsc'
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

# In[23]:


# Set path to your data file
file_path = "../data/Category_2.csv"

# Load and clean data
properties = load_data(file_path)


# ## 4. Examine the Data
# 
# Let's examine the data to understand what we're working with.

# In[24]:


# Display a few sample rows
properties.head()


# In[25]:


# Check data types and missing values
properties.info()


# In[26]:


# Look at statistics for numeric columns
properties.describe()


# In[27]:


# Check which ZIP codes are present
if 'SiteZIP' in properties.columns:
    print("ZIP code counts:")
    print(properties['SiteZIP'].value_counts())


# In[28]:


# Check distribution of property types
if 'LandUseDsc' in properties.columns:
    print("Land use types:")
    print(properties['LandUseDsc'].value_counts().head(10))


# ## 5. Filter Functions
# 
# Define functions to filter properties based on our criteria.

# In[29]:


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

# In[30]:


# Apply Category 2 filters
cat2_properties = filter_category2(properties)


# Display the first few properties that match
if len(cat2_properties) > 0:
    # Select a subset of columns to display
    display_columns = [
        'SiteAddr', 'SiteCity', 'SiteZIP', 'BathTtlCt', 'BedCt',
        'BldgSqFt', 'LotSqFt', 'YrBlt', 'MktTtlVal', 'LandUseDsc', 'ZoneCd'
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

# In[31]:


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

# In[32]:


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
    Calculate basement space score for Category 2 (10 points max)
    For conversion-ready properties, unfinished basement space is preferred
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    bsmt_fin = row['BsmtFinSqFt'] if pd.notna(row['BsmtFinSqFt']) else 0
    bsmt_unfin = row['BsmtUnFinSqFt'] if pd.notna(row['BsmtUnFinSqFt']) else 0
    
    # For conversion-ready properties, unfinished space is preferred
    if bsmt_unfin > 800:
        return 10  # Large unfinished basement (ideal for conversion)
    elif bsmt_unfin >= 500:
        return 8   # Good unfinished space for conversion
    elif bsmt_unfin >= 300 and bsmt_fin >= 300:
        return 6   # Mix of finished/unfinished space
    elif bsmt_fin > 500:
        return 4   # Already finished (less flexible for conversion)
    elif (bsmt_fin + bsmt_unfin) > 0:
        return 2   # Some basement is better than none
    else:
        return 0   # No basement




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
    Calculate condition score for Category 2 (10 points max)
    For conversion-ready properties, "Average" condition scores highest (easier/cheaper to convert)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['Condition']):
        condition = str(row['Condition']).strip().lower()
        if condition == 'average':
            return 10  # Ideal for conversion (not too dilapidated, not too finished)
        elif condition == 'fair':
            return 8   # Can be improved but may need more work
        elif condition == 'good':
            return 6   # May be over-improved for conversion purposes
        elif condition == 'excellent':
            return 4   # Likely over-improved, conversion might "waste" existing finishes
        else:
            return 2   # Poor condition may indicate deeper issues
    return 0




def calculate_year_built_scorecat2(row):
    """
    Calculate year built score for Category 2 (10 points max)
    For conversion-ready properties, 1950-1980 era homes score highest
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['YrBlt']):
        if 1950 <= row['YrBlt'] <= 1980:
            return 10  # Ideal era for conversion-friendly layouts
        elif 1940 <= row['YrBlt'] < 1950:
            return 8   # Post-war homes, often good candidates
        elif 1980 < row['YrBlt'] <= 2000:
            return 6   # Newer but still adaptable
        elif row['YrBlt'] > 2000:
            return 4   # Very new homes (less need for conversion)
        else:
            return 2   # Very old homes (may have structural challenges)
    return 0


def calculate_zoning_scorecat2(row):
    """
    Calculate zoning score for Category 2 (10 points max)
    Evaluates zoning compatibility with rental unit conversion
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['ZoneCd']):
        zone_code = str(row['ZoneCd']).strip()
        
        # Multi-family zones score highest
        if any(code in zone_code for code in ['LR1', 'LR2', 'LR3']):
            return 10  # Lowrise multi-family zones - ideal for conversion
        
        # Residential small lot is good for ADUs
        elif 'RSL' in zone_code:
            return 9  # Residential Small Lot - designed for higher density
            
        # Higher density residential zones
        elif any(code in zone_code for code in ['R-18', 'R-24', 'NR3']):
            return 8  # Higher density residential zones
            
        # Medium density residential
        elif any(code in zone_code for code in ['NR2']):
            return 6  # Medium density residential
            
        # Lower density residential
        elif any(code in zone_code for code in ['R-6']):
            return 4  # Lower density residential
            
        # Other residential zones not clearly categorized
        elif 'R-' in zone_code:
            return 3  # Other residential zones
            
        # Non-residential or industrial zones
        elif 'UI' in zone_code:
            return 0  # Urban Industrial - not suitable for residential conversion
            
        else:
            return 2  # Unknown zoning
    
    return 0  # No zoning information available


# In[33]:


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
        'YrBlt': 1985,
        'ZoneCd' : 'Residential Small Lot'
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
    print(f"Zoning ScoreCat2: {calculate_zoning_scorecat2(sample_property)}")
    
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
        calculate_year_built_scorecat2(sample_property) +
        calculate_zoning_scorecat2(sample_property)
    )
    
    print(f"Total Desirability ScoreCat2: {total_scorecat2}")




# Run the test to see if our scoring functions work correctly
test_desirability_scoring_functions()


# ## 9. Calculate Cat2 Seller Likelihood
# 
# Determine seller likelihood for all filtered properties.

# In[43]:


def calculate_seller_likelihood_score(properties_df):
    """
    Calculate the likelihood of sellers being willing to sell their properties,
    with specific emphasis on factors relevant to conversion-ready property owners.
    
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
    
    # Ensure numeric conversion for key financial columns
    financial_columns = ['MktTtlVal', 'AssdTtlVal', 'TaxTtl1', 'TaxTtl2', 'TaxTtl3']
    for col in financial_columns:
        if col in df.columns:
            # Remove currency symbols and commas, then convert to numeric
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
                errors='coerce'
            )
    
    # ===== OWNERSHIP FACTORS =====
    
    # Length of ownership - properties owned longer may be more likely to sell
    if 'DocRcrdgDt_County' in df.columns:
        df['OwnershipLength'] = pd.Timestamp.now().year - pd.to_datetime(df['DocRcrdgDt_County'], errors='coerce').dt.year
        
        # Apply scores based on ownership length (higher for conversion-ready)
        df.loc[df['OwnershipLength'] >= 15, 'SellerLikelihoodScore'] += 20
        df.loc[(df['OwnershipLength'] >= 7) & (df['OwnershipLength'] < 15), 'SellerLikelihoodScore'] += 15
        df.loc[(df['OwnershipLength'] >= 3) & (df['OwnershipLength'] < 7), 'SellerLikelihoodScore'] += 7
        
        # Add factor descriptions
        df.loc[df['OwnershipLength'] >= 15, 'SellerLikelihoodFactors'] += 'Very long-term owner; '
        df.loc[(df['OwnershipLength'] >= 7) & (df['OwnershipLength'] < 15), 'SellerLikelihoodFactors'] += 'Long-term owner; '
    
    # Non-owner occupied properties may be more likely to sell
    if 'OwnerOccupiedInd' in df.columns:
        df.loc[df['OwnerOccupiedInd'] == False, 'SellerLikelihoodScore'] += 15
        df.loc[df['OwnerOccupiedInd'] == False, 'SellerLikelihoodFactors'] += 'Investment property; '
    
    # Corporate ownership might indicate investment property
    if 'OwnerCorporateInd' in df.columns:
        df.loc[df['OwnerCorporateInd'] == True, 'SellerLikelihoodScore'] += 12
        df.loc[df['OwnerCorporateInd'] == True, 'SellerLikelihoodFactors'] += 'Corporate owner; '
    
    # Different mailing address may indicate non-local owner
    if all(col in df.columns for col in ['OwnerAddr', 'SiteAddr']):
        df['DifferentMailingAddr'] = df['OwnerAddr'] != df['SiteAddr']
        df.loc[df['DifferentMailingAddr'], 'SellerLikelihoodScore'] += 12
        df.loc[df['DifferentMailingAddr'], 'SellerLikelihoodFactors'] += 'Non-local owner; '
    
    # ===== FINANCIAL FACTORS =====
    
    # Tax assessment increases - particularly important for conversion-ready properties
    if all(col in df.columns for col in ['TaxTtl1', 'TaxTtl2', 'TaxTtl3']):
        # Calculate year-over-year tax increases as percentage
        df['TaxChange_Recent'] = ((df['TaxTtl1'] - df['TaxTtl2']) / df['TaxTtl2'] * 100).fillna(0)
        df['TaxChange_Previous'] = ((df['TaxTtl2'] - df['TaxTtl3']) / df['TaxTtl3'] * 100).fillna(0)
        
        # Score based on tax increases (higher weight for conversion-ready)
        df.loc[df['TaxChange_Recent'] >= 15, 'SellerLikelihoodScore'] += 18
        df.loc[(df['TaxChange_Recent'] >= 10) & (df['TaxChange_Recent'] < 15), 'SellerLikelihoodScore'] += 12
        df.loc[(df['TaxChange_Recent'] >= 5) & (df['TaxChange_Recent'] < 10), 'SellerLikelihoodScore'] += 6
        
        # Consistent tax increases over multiple years
        df.loc[(df['TaxChange_Recent'] >= 8) & (df['TaxChange_Previous'] >= 8), 'SellerLikelihoodScore'] += 10
        
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
        # Safely calculate value assessment ratio
        df['ValueAssessmentRatio'] = df.apply(
            lambda row: row['MktTtlVal'] / row['AssdTtlVal'] if row['AssdTtlVal'] > 0 else 1, 
            axis=1
        )
        
        df.loc[df['ValueAssessmentRatio'] > 1.25, 'SellerLikelihoodScore'] += 12
        df.loc[df['ValueAssessmentRatio'] > 1.25, 'SellerLikelihoodFactors'] += 'Significant recent appreciation; '
    
    # Length of ownership combined with age may indicate life transition
    if 'OwnershipLength' in df.columns:
        # Very long ownership might indicate aging owners considering downsizing
        df.loc[df['OwnershipLength'] >= 25, 'SellerLikelihoodScore'] += 8
        df.loc[df['OwnershipLength'] >= 25, 'SellerLikelihoodFactors'] += 'Potential life transition; '
    
    # ===== SCORE NORMALIZATION =====
    
    # Calculate the maximum possible score
    max_possible_score = 117  # Sum of all maximum points from factors above
    
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


# ## 10. Calculate ScoreCat2s for All Properties
# 
# Apply the scoring functions to all filtered properties.

# In[44]:


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
    properties['PropertyTypeScorecat2'] = properties.apply(calculate_property_type_scorecat2, axis=1)
    properties['BathroomDistributionScorecat2'] = properties.apply(calculate_bathroom_distribution_scorecat2, axis=1)
    properties['BuildingSizeScorecat2'] = properties.apply(calculate_building_size_scorecat2, axis=1)
    properties['BasementSpaceScorecat2'] = properties.apply(calculate_basement_space_scorecat2, axis=1)
    properties['StoriesCountScorecat2'] = properties.apply(calculate_stories_count_scorecat2, axis=1)
    properties['ZipCodeValueScorecat2'] = properties.apply(calculate_zip_code_value_scorecat2, axis=1)
    properties['LotSizeScorecat2'] = properties.apply(calculate_lot_size_scorecat2, axis=1)
    properties['ConditionScorecat2'] = properties.apply(calculate_condition_scorecat2, axis=1)
    properties['YearBuiltScorecat2'] = properties.apply(calculate_year_built_scorecat2, axis=1)
    
    return properties
    
def calculate_combined_scorecat2s(properties):
    """
    Calculate total and combined scores, and assign priority tiers
    
    Args:
        properties (pandas.DataFrame): Property data with individual scores
        
    Returns:
        pandas.DataFrame: Property data with total scores and priority tiers
    """
    # Calculate desirability score
    desirability_columns = [
        'PropertyTypeScoreCat2', 'BathroomDistributionScoreCat2', 'BuildingSizeScoreCat2',
        'BasementSpaceScoreCat2', 'StoriesCountScoreCat2', 'ZipCodeValueScoreCat2',
        'LotSizeScoreCat2', 'ConditionScoreCat2', 'YearBuiltScoreCat2', 'ZoningScoreCat2'
    ]
    
    properties['DesirabilityScoreCat2'] = properties[desirability_columns].sum(axis=1)
    
    # Calculate seller likelihood score (this already normalizes to 0-100)
    properties = calculate_seller_likelihood_score(properties)
    
    # Store the raw scores for reference
    properties['DesirabilityScoreCat2_Raw'] = properties['DesirabilityScoreCat2']
    properties['SellerLikelihoodScore_Raw'] = properties['SellerLikelihoodScore']
    
    # Normalize desirability score to 0-100 scale
    max_desirability = 100  # Sum of all max points from desirability factors
    properties['DesirabilityScoreCat2'] = (properties['DesirabilityScoreCat2'] / max_desirability) * 100
    
    # Calculate final combined score (weighted average of normalized scores)
    properties['FinalScoreCat2'] = (
        properties['DesirabilityScoreCat2'] * 0.7 + 
        properties['SellerLikelihoodScore'] * 0.3
    )
    
    # Assign priority tiers based on the combined final score
    def assign_tier(score):
        # Working with normalized scores now (0-100)
        if score >= 80:  # 80% or higher
            return "Tier 1"
        elif score >= 65:  # 65-80%
            return "Tier 2"
        elif score >= 50:  # 50-65%
            return "Tier 3"
        else:  # Below 50%
            return "Tier 4"
    
    properties['PriorityTier'] = properties['FinalScoreCat2'].apply(assign_tier)
    
    print("Score ranges for each tier:")
    print(f"  Tier 1: 80.0-100.0")
    print(f"  Tier 2: 65.0-80.0")
    print(f"  Tier 3: 50.0-65.0")
    print(f"  Tier 4: 0-50.0")
    
    return properties


# In[45]:


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
        'SiteAddrj', 'SiteCity', 'SiteZIP', 'BathTtlCt', 'BedCt',
        'BldgSqFt', 'YrBlt', 'MktTtlVal', 'DesirabilityScoreCat2', 
        'SellerLikelihoodScore', 'FinalScoreCat2', 'PriorityTier', 'ZoneCd'
    ]
    
    # Make sure all requested columns exist
    display_columns = [col for col in display_columns if col in cat2_properties.columns]
    
    print("\nTop 5 properties by final score:")
    top_properties = cat2_properties.sort_values('FinalScoreCat2', ascending=False)
    display(top_properties[display_columns].head())


# ## 11.  Reorder Results

# In[ ]:


import pandas as pd

# Assuming you already have your DataFrame loaded with the property data and scores
# For example: df = pd.read_csv('properties_with_scores.csv')

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
        'PriorityTier', 'DesirabilityScoreCat2', 'SellerLikelihoodScore', 'FinalScoreCat2'
    ]
    
    # Add factor score columns based on category
    if category == 2:  # Conversion-Ready Properties
        factor_cols = [
            'PropertyTypeScoreCat2', 'BathroomDistributionScoreCat2', 'BuildingSizeScoreCat2',
            'BasementSpaceScoreCat2', 'StoriesCountScoreCat2', 'ZipCodeValueScoreCat2',
            'LotSizeScoreCat2', 'ConditionScoreCat2', 'YearBuiltScoreCat2', 'ZoningScoreCat2'
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
    if category == 2:  # Conversion-Ready Properties
        category_cols = [
            'BsmtFinSqFt', 'BsmtUnFinSqFt', 'ZoneCd', 'ZoneDsc', 'StoriesCt'
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
# Save the filtered and scorecat2d properties to a CSV file for further analysis.

# In[18]:


# Save the results to a CSV file
if len(cat2_properties) > 0:
    # Reorder the columns for better readability
    cat2_properties_reordered = reorder_columns(cat2_properties, 2)
    
    # Create output directory if it doesn't exist
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV - use the reordered dataframe
    output_file = f"{output_dir}/category2_properties_conversion_ready.csv"
    cat2_properties_reordered.to_csv(output_file, index=False)
    print(f"Saved {len(cat2_properties)} category 2 conversion ready properties to {output_file}")


# In[ ]:




