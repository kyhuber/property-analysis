#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Property Analysis for Category 1 (Rent-Ready Properties)
--------------------------------------------------------
This script analyzes property data to identify properties that fit Category 1
criteria (already configured with separate rentable units) and ranks them based
on desirability and seller likelihood.

Author: Claude AI
Date: April 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import re

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

def main():
    """Main function to run the property analysis"""
    
    # Load data
    print("Loading property data...")
    properties = load_data()
    
    # Apply filters for Category 1
    print("Applying Category 1 filters...")
    cat1_properties = filter_category1(properties)
    
    # Calculate neighborhood averages for value ratio calculation
    print("Calculating neighborhood averages...")
    zip_averages = calculate_zip_averages(cat1_properties)
    cat1_properties = cat1_properties.merge(zip_averages, on='SiteZIP', how='left')
    
    # Calculate all scores
    print("Calculating property scores...")
    cat1_properties = calculate_all_scores(cat1_properties)
    
    # Calculate total and combined scores
    print("Calculating combined scores and assigning tiers...")
    cat1_properties = calculate_combined_scores(cat1_properties)
    
    # Sort properties by priority tier and score
    sorted_properties = cat1_properties.sort_values(
        by=['PriorityTier', 'CombinedScore'], 
        ascending=[True, False]
    )
    
    # Save results
    output_directory = 'output'
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Saving results to {output_directory}...")
    sorted_properties.to_csv(f'{output_directory}/category1_ranked_properties.csv', index=False)
    
    # Generate visualizations
    print("Generating visualizations...")
    create_visualizations(sorted_properties, output_directory)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print_summary_stats(sorted_properties)
    
    return sorted_properties


def load_data(file_path=None):
    """
    Load property data from CSV file
    
    Args:
        file_path (str): Path to the CSV file, if None, prompts user for path
        
    Returns:
        pandas.DataFrame: Property data
    """
    if file_path is None:
        file_path = input("Enter the path to your property data CSV file: ")
    
    # Load the data
    properties = pd.read_csv(file_path)
    
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
        'MktTtlVal', 'YrBlt', 'StoriesCt', 'Acres'
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
    
    print(f"Loaded {len(properties)} properties")
    return properties


def filter_category1(properties):
    """
    Apply Category 1 filters to properties
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Filtered property data
    """
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
    
    print(f"Found {len(cat1_properties)} properties matching Category 1 criteria")
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
    
    return zip_averages


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


#========== SELLER LIKELIHOOD SCORE FUNCTIONS ==========

def calculate_ownership_duration_score(row):
    """
    Calculate ownership duration score (30 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-30
    """
    if pd.notna(row['DocRcrdgDt_County']):
        try:
            # Calculate years owned
            current_date = pd.Timestamp.now()
            record_date = row['DocRcrdgDt_County']
            years_owned = (current_date - record_date).days / 365.25
            
            if 5 <= years_owned <= 7:
                return 30
            elif 8 <= years_owned <= 12:
                return 25
            elif 13 <= years_owned <= 20:
                return 20
            elif years_owned > 20:
                return 15
            elif 3 <= years_owned <= 4:
                return 10
            else:
                return 5
        except:
            return 5  # Default if calculation fails
    return 5  # Default if date is missing


def calculate_owner_occupancy_score(row):
    """
    Calculate owner occupancy score (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-15
    """
    if pd.notna(row['OwnerOccupiedInd']):
        if row['OwnerOccupiedInd'] == False:
            return 15
        else:
            return 5
    return 5  # Default if data is missing


def calculate_owner_location_score(row):
    """
    Calculate owner location score (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-15
    """
    site_zip = str(row['SiteZIP']) if pd.notna(row['SiteZIP']) else ''
    owner_zip = str(row['OwnerZIP']) if pd.notna(row['OwnerZIP']) else ''
    site_state = str(row['SiteState']) if pd.notna(row['SiteState']) else ''
    owner_state = str(row['OwnerState']) if pd.notna(row['OwnerState']) else ''
    
    if site_zip != owner_zip and site_state != owner_state:
        return 15  # Out of state owner
    elif site_zip != owner_zip:
        return 10  # Different ZIP, same state
    else:
        return 3  # Same ZIP
    

def calculate_tax_trend_score(row):
    """
    Calculate tax trend score (15 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-15
    """
    tax1 = row['TaxTtl1'] if pd.notna(row['TaxTtl1']) else 0
    tax2 = row['TaxTtl2'] if pd.notna(row['TaxTtl2']) else 0
    
    if tax2 == 0:
        return 8  # Can't calculate trend, assign middle value
    
    tax_increase = (tax1 - tax2) / tax2 * 100
    
    if tax_increase > 10:
        return 15
    elif tax_increase >= 5:
        return 12
    elif tax_increase >= 0:
        return 8
    else:
        return 5


def calculate_value_ratio_score(row):
    """
    Calculate value ratio score (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['MktTtlVal']) and pd.notna(row['ZipAvgValue']) and row['ZipAvgValue'] > 0:
        value_ratio = (row['MktTtlVal'] / row['ZipAvgValue']) * 100
        
        if value_ratio < 80:
            return 10
        elif value_ratio < 90:
            return 8
        elif value_ratio <= 110:
            return 5
        else:
            return 3
    return 5  # Default if data is missing


def calculate_investment_potential_score(row):
    """
    Calculate investment potential score (10 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-10
    """
    if pd.notna(row['InvestmentProp']) and row['InvestmentProp'] == True:
        return 10
    elif pd.notna(row['LandUseDsc']) and "Duplex" in str(row['LandUseDsc']):
        return 8
    else:
        return 5


def calculate_bank_owned_score(row):
    """
    Calculate bank owned score (5 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-5
    """
    if pd.notna(row['BankOwnedInd']) and row['BankOwnedInd'] == True:
        return 5
    else:
        return 0


def calculate_corporate_owned_score(row):
    """
    Calculate corporate owned score (5 points max)
    
    Args:
        row (pandas.Series): Row of property data
        
    Returns:
        int: Score between 0-5
    """
    if pd.notna(row['OwnerCorporateInd']) and row['OwnerCorporateInd'] == True:
        return 5
    else:
        return 0


def calculate_all_scores(properties):
    """
    Calculate all individual scores for properties
    
    Args:
        properties (pandas.DataFrame): Property data
        
    Returns:
        pandas.DataFrame: Property data with scores
    """
    # Calculate desirability scores
    properties['PropertyTypeScore'] = properties.apply(calculate_property_type_score, axis=1)
    properties['BathroomDistributionScore'] = properties.apply(calculate_bathroom_distribution_score, axis=1)
    properties['BuildingSizeScore'] = properties.apply(calculate_building_size_score, axis=1)
    properties['BasementSpaceScore'] = properties.apply(calculate_basement_space_score, axis=1)
    properties['StoriesCountScore'] = properties.apply(calculate_stories_count_score, axis=1)
    properties['ZipCodeValueScore'] = properties.apply(calculate_zip_code_value_score, axis=1)
    properties['LotSizeScore'] = properties.apply(calculate_lot_size_score, axis=1)
    properties['ConditionScore'] = properties.apply(calculate_condition_score, axis=1)
    properties['YearBuiltScore'] = properties.apply(calculate_year_built_score, axis=1)
    
    # Calculate seller likelihood scores
    properties['OwnershipDurationScore'] = properties.apply(calculate_ownership_duration_score, axis=1)
    properties['OwnerOccupancyScore'] = properties.apply(calculate_owner_occupancy_score, axis=1)
    properties['OwnerLocationScore'] = properties.apply(calculate_owner_location_score, axis=1)
    properties['TaxTrendScore'] = properties.apply(calculate_tax_trend_score, axis=1)
    properties['ValueRatioScore'] = properties.apply(calculate_value_ratio_score, axis=1)
    properties['InvestmentPotentialScore'] = properties.apply(calculate_investment_potential_score, axis=1)
    properties['BankOwnedScore'] = properties.apply(calculate_bank_owned_score, axis=1)
    properties['CorporateOwnedScore'] = properties.apply(calculate_corporate_owned_score, axis=1)
    
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
    
    # Calculate seller likelihood score
    seller_likelihood_columns = [
        'OwnershipDurationScore', 'OwnerOccupancyScore', 'OwnerLocationScore',
        'TaxTrendScore', 'ValueRatioScore', 'InvestmentPotentialScore',
        'BankOwnedScore', 'CorporateOwnedScore'
    ]
    
    properties['SellerLikelihoodScore'] = properties[seller_likelihood_columns].sum(axis=1)
    
    # Calculate combined score
    properties['CombinedScore'] = (properties['DesirabilityScore'] * 0.6) + \
                                 (properties['SellerLikelihoodScore'] * 0.4)
    
    # Assign priority tiers
    properties['PriorityTier'] = properties['CombinedScore'].apply(assign_tier)
    
    return properties


def assign_tier(score):
    """
    Assign priority tier based on combined score
    
    Args:
        score (float): Combined score
        
    Returns:
        str: Priority tier
    """
    if score >= 80:
        return "Tier 1"
    elif score >= 65:
        return "Tier 2"
    elif score >= 50:
        return "Tier 3"
    else:
        return "Tier 4"


def create_visualizations(properties, output_directory):
    """
    Create visualizations of property analysis results
    
    Args:
        properties (pandas.DataFrame): Property data with scores
        output_directory (str): Directory to save visualizations
    """
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Priority tier distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=properties, x='PriorityTier', order=["Tier 1", "Tier 2", "Tier 3", "Tier 4"])
    plt.title('Number of Properties by Priority Tier', fontsize=16)
    plt.xlabel('Priority Tier', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_directory}/tier_distribution.png')
    
    # 2. Score distribution by ZIP code
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=properties, x='SiteZIP', y='CombinedScore')
    plt.title('Combined Score Distribution by ZIP Code', fontsize=16)
    plt.xlabel('ZIP Code', fontsize=12)
    plt.ylabel('Combined Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_directory}/score_by_zip.png')
    
    # 3. Desirability vs. Seller Likelihood scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=properties, 
        x='DesirabilityScore', 
        y='SellerLikelihoodScore',
        hue='PriorityTier',
        palette=['green', 'blue', 'orange', 'red'],
        alpha=0.7
    )
    plt.title('Desirability vs. Seller Likelihood', fontsize=16)
    plt.xlabel('Desirability Score', fontsize=12)
    plt.ylabel('Seller Likelihood Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_directory}/desirability_vs_likelihood.png')
    
    # 4. Top factors correlation heatmap
    score_columns = [
        'PropertyTypeScore', 'BathroomDistributionScore', 'BuildingSizeScore',
        'ZipCodeValueScore', 'ConditionScore', 'YearBuiltScore',
        'OwnershipDurationScore', 'OwnerOccupancyScore', 'OwnerLocationScore',
        'TaxTrendScore', 'ValueRatioScore', 'CombinedScore'
    ]
    
    plt.figure(figsize=(14, 10))
    correlation = properties[score_columns].corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
    plt.title('Correlation Between Scoring Factors', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_directory}/score_correlations.png')


def print_summary_stats(properties):
    """
    Print summary statistics of property analysis results
    
    Args:
        properties (pandas.DataFrame): Property data with scores
    """
    # Count properties by tier
    tier_counts = properties['PriorityTier'].value_counts().sort_index()
    print("Properties by Priority Tier:")
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count}")
    
    # Count properties by ZIP code
    zip_counts = properties['SiteZIP'].value_counts().sort_index()
    print("\nProperties by ZIP Code:")
    for zip_code, count in zip_counts.items():
        print(f"  {zip_code}: {count}")
    
    # Average scores
    print("\nAverage Scores:")
    print(f"  Desirability Score: {properties['DesirabilityScore'].mean():.2f}")
    print(f"  Seller Likelihood Score: {properties['SellerLikelihoodScore'].mean():.2f}")
    print(f"  Combined Score: {properties['CombinedScore'].mean():.2f}")
    
    # Top 10 properties
    print("\nTop 10 Properties:")
    top_properties = properties.sort_values('CombinedScore', ascending=False).head(10)
    for i, (_, prop) in enumerate(top_properties.iterrows(), 1):
        print(f"  {i}. {prop['SiteAddr']}, {prop['SiteCity']} - Score: {prop['CombinedScore']:.2f}")


if __name__ == "__main__":
    top_properties = main()
    
    # Display top 20 properties
    print("\nTop 20 Properties:")
    display_columns = [
        'ParcelId', 'SiteAddr', 'SiteCity', 'SiteZIP', 'BathTtlCt', 'BedCt',
        'BldgSqFt', 'YrBlt', 'MktTtlVal', 'DesirabilityScore', 
        'SellerLikelihoodScore', 'CombinedScore', 'PriorityTier'
    ]
    
    top_20 = top_properties.sort_values('CombinedScore', ascending=False)[display_columns].head(20)
    print(top_20)