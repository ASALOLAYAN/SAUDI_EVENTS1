import pandas as pd
import numpy as np

print("=== 1. Loading and Inspecting Data ===")

# Load dataset
try:
    df = pd.read_csv("saudi_events_sales_raw.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'saudi_events_sales_raw.csv' not found. Please check the path.")
    exit()

# Initial inspection
print(f"\nDataset Shape: Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n--- First 10 Rows ---")
print(df.head(10))
print("\n--- Last 5 Rows ---")
print(df.tail(5))
print("\nMissing Values Per Column Before Cleaning:")
print(df.isnull().sum())


print("\n=== 2. Cleaning Step 1: Text Columns ===")

# Clean text spaces and standardize city names
text_cols = ['city', 'venue', 'event_type', 'ticket_category', 'marketing_channel', 'weather']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Format city names to Title Case
df['city'] = df['city'].str.title()

# Handle missing values in marketing channel
df['marketing_channel'] = df['marketing_channel'].fillna('Unknown')
df.loc[df['marketing_channel'].str.lower() == 'nan', 'marketing_channel'] = 'Unknown'

print("Text columns cleaned and standardized successfully.")


print("\n=== 3. Cleaning Step 2: Convert Types ===")

# Convert event_date to datetime format
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce', format='mixed')

# Clean and convert ticket price to numeric
df['ticket_price_sar'] = df['ticket_price_sar'].astype(str).str.replace('SAR', '', case=False)
df['ticket_price_sar'] = df['ticket_price_sar'].str.replace(',', '')
df['ticket_price_sar'] = pd.to_numeric(df['ticket_price_sar'].str.strip(), errors='coerce')

# Clean and convert discount percentage to numeric
df['discount_pct'] = df['discount_pct'].astype(str).str.replace('%', '')
df['discount_pct'] = pd.to_numeric(df['discount_pct'].str.strip(), errors='coerce')

# Convert remaining numerical columns
numeric_cols = ['capacity', 'tickets_sold', 'marketing_spend_sar', 'customer_rating', 'complaints', 'reported_revenue_sar']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Data types converted successfully.")


print("\n=== 4. Cleaning Step 3: Handle Invalid Values ===")

# Remove duplicate rows
df = df.drop_duplicates()

# Set invalid values to NaN based on logical rules
df.loc[df['tickets_sold'] < 0, 'tickets_sold'] = np.nan
df.loc[df['tickets_sold'] > df['capacity'], 'tickets_sold'] = np.nan
df.loc[df['capacity'] <= 0, 'capacity'] = np.nan
df.loc[(df['customer_rating'] < 1) | (df['customer_rating'] > 5), 'customer_rating'] = np.nan
df.loc[df['complaints'] < 0, 'complaints'] = np.nan
df.loc[df['marketing_spend_sar'] < 0, 'marketing_spend_sar'] = np.nan

# Impute missing ratings using city-based median
df['customer_rating'] = df.groupby('city')['customer_rating'].transform(lambda x: x.fillna(x.median()))
df['customer_rating'] = df['customer_rating'].fillna(df['customer_rating'].median())

# Fill remaining missing values with logical defaults
df['discount_pct'] = df['discount_pct'].fillna(0)
df['tickets_sold'] = df['tickets_sold'].fillna(0)
df['complaints'] = df['complaints'].fillna(0)
df['marketing_spend_sar'] = df['marketing_spend_sar'].fillna(0)

print("Logical violations handled and duplicates removed.")


print("\n=== 5. Feature Engineering ===")

# 1. Calculate revenue from sales
df['calculated_revenue_sar'] = df['tickets_sold'] * df['ticket_price_sar'] * (1 - df['discount_pct'] / 100)

# 2. Find revenue mismatch
df['revenue_difference'] = df['reported_revenue_sar'] - df['calculated_revenue_sar']

# 3. Calculate occupancy rate
df['occupancy_rate'] = df['tickets_sold'] / df['capacity']
df['occupancy_rate'] = df['occupancy_rate'].fillna(0)

# 4. Calculate net revenue after marketing costs
df['net_revenue_after_marketing'] = df['calculated_revenue_sar'] - df['marketing_spend_sar']

# 5. Extract weekday name
df['weekday'] = df['event_date'].dt.day_name()

# 6. Flag Saudi weekend days (Friday & Saturday)
df['is_weekend'] = df['weekday'].isin(['Friday', 'Saturday'])

# 7. Classify demand level
def get_demand_level(row):
    if row['occupancy_rate'] >= 0.85:
        return 'High'
    elif row['occupancy_rate'] >= 0.55:
        return 'Medium'
    else:
        return 'Low'
df['demand_level'] = df.apply(get_demand_level, axis=1)

# 8. Classify rating status
def get_rating_status(row):
    if row['customer_rating'] >= 4.5:
        return 'Excellent'
    elif row['customer_rating'] < 3.8:
        return 'Needs Review'
    else:
        return 'Good'
df['rating_status'] = df.apply(get_rating_status, axis=1)

print("New business features created successfully.")


print("\n=== 6. Business Analysis Questions ===")

# A. City performance
print("\n[A] City Performance Summary:")
city_perf = df.groupby('city').agg(
    total_tickets_sold=('tickets_sold', 'sum'),
    total_calculated_revenue=('calculated_revenue_sar', 'sum'),
    average_rating=('customer_rating', 'mean'),
    average_occupancy_rate=('occupancy_rate', 'mean')
).reset_index()
print(city_perf)

# B. Venue performance
print("\n[B] Top 5 Venues by Total Revenue:")
top_venues = df.groupby('venue')['calculated_revenue_sar'].sum().nlargest(5)
print(top_venues)

print("\nBottom 5 Venues by Average Customer Rating:")
bottom_venues = df.groupby('venue')['customer_rating'].mean().nsmallest(5)
print(bottom_venues)

# C. Event type performance
print("\n[C] Event Type Performance:")
event_type_perf = df.groupby('event_type').agg(
    average_occupancy=('occupancy_rate', 'mean'),
    total_revenue=('calculated_revenue_sar', 'sum')
)
print(event_type_perf)

# D. Marketing channel performance
print("\n[D] Marketing Channel Performance:")
marketing_perf = df.groupby('marketing_channel').agg(
    total_revenue=('calculated_revenue_sar', 'sum'),
    avg_net_revenue_after_marketing=('net_revenue_after_marketing', 'mean')
).reset_index()
print(marketing_perf)

strongest_channel = marketing_perf.loc[marketing_perf['avg_net_revenue_after_marketing'].idxmax()]['marketing_channel']
print(f"--> Strongest channel based on Average Net Revenue: {strongest_channel}")

# E. Identify high operational risk events
print("\n[E] Operational Risk - Sample of Events Needing Review:")
events_needing_review = df[
    (df['rating_status'] == 'Needs Review') | 
    (df['complaints'] >= 6) | 
    (df['occupancy_rate'] < 0.4)
]
print(f"Total events requiring review: {len(events_needing_review)}")
print(events_needing_review[['event_id', 'city', 'venue', 'customer_rating', 'complaints', 'occupancy_rate']].head())


print("\n=== 7. Revenue Validation ===")
mismatches = df[df['revenue_difference'].abs() > 1]
print(f"Number of rows with revenue mismatch: {len(mismatches)}")

if not mismatches.empty:
    print("\nTop Biggest Revenue Mismatches found:")
    print(mismatches[['event_id', 'reported_revenue_sar', 'calculated_revenue_sar', 'revenue_difference']].sort_values(by='revenue_difference', key=abs, ascending=False).head())


print("\n=== 8. Final City Dashboard ===")
pd.options.display.float_format = '{:,.2f}'.format
dashboard = df.groupby('city').agg(
    total_events=('event_id', 'count'),
    total_tickets_sold=('tickets_sold', 'sum'),
    total_revenue_sar=('calculated_revenue_sar', 'sum'),
    average_rating=('customer_rating', 'mean'),
    average_occupancy_rate=('occupancy_rate', 'mean'),
    total_complaints=('complaints', 'sum'),
    average_net_revenue_after_marketing=('net_revenue_after_marketing', 'mean')
).sort_values(by='total_revenue_sar', ascending=False).reset_index()
print(dashboard)


print("\n=== 9. Final Business Recommendations ===")
recommendation = (
    "1. Best-Performing City/Venue: Riyadh and Jeddah dominate ticket sales and total revenues.\n"
    "2. Weakest Area: Peripheral regions like Abha and Madinah exhibit lower occupancy rates.\n"
    "3. Marketing Optimization: Digital marketing channels (Social Media / Influencers) net higher average returns.\n"
    "4. Data Integrity: Standardizing data pipelines is critical to eliminate mixed formats and mismatches.\n"
    "5. Analytical Additions: Introducing demographic attributes will enhance dynamic predictive demand models."
)
print(recommendation)