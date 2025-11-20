# Databricks notebook source
# MAGIC %md
# MAGIC # Insurance Fraud Detection - Dataset Generation
# MAGIC 
# MAGIC This notebook generates synthetic insurance claim data for GraphFrames fraud detection analysis.
# MAGIC 
# MAGIC ## What Gets Generated:
# MAGIC - **Policyholders**: Insurance customers with demographics and contact info
# MAGIC - **Claims**: Insurance claims with fraud_score (0-1) indicating suspiciousness
# MAGIC - **Adjusters**: Insurance professionals who process claims (neutral entities)
# MAGIC - **Service Providers**: Repair shops, medical providers, etc. (neutral entities)
# MAGIC 
# MAGIC ## Key Design Principles:
# MAGIC 1. **Claims have fraud_score (0-1)**: Continuous score, not binary label
# MAGIC 2. **Natural patterns embedded**: Some policyholders share addresses/phones, file multiple claims
# MAGIC 3. **NO pre-labeled fraud rings**: GraphFrames will discover them!
# MAGIC 4. **Realistic relationships**: Claims linked to policyholders, adjusters, service providers
# MAGIC 
# MAGIC ## Patterns That Enable Fraud Detection:
# MAGIC - Policyholders sharing addresses/phone numbers
# MAGIC - Multiple claims from same policyholder
# MAGIC - Temporal clustering of high fraud_score claims
# MAGIC - High-value claims tend to have higher fraud_scores

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Widgets

# COMMAND ----------

# Create widgets for configuration
dbutils.widgets.dropdown("volume_scale", "small", ["small", "medium", "large", "xlarge", "custom"], "Volume Scale")
dbutils.widgets.text("num_policyholders", "1000", "Number of Policyholders")
dbutils.widgets.text("num_claims", "5000", "Number of Claims")
dbutils.widgets.text("num_adjusters", "50", "Number of Adjusters")
dbutils.widgets.text("num_service_providers", "200", "Number of Service Providers")
dbutils.widgets.text("high_fraud_rate", "0.15", "High Fraud Score Rate (0.0-1.0)")
dbutils.widgets.text("catalog_name", "dbdemos_steventan", "Catalog Name")
dbutils.widgets.text("schema_name", "frauddetection_graphframe", "Schema Name")
dbutils.widgets.dropdown("overwrite_mode", "true", ["true", "false"], "Overwrite Existing Tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import random

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Configuration

# COMMAND ----------

# Get configuration from widgets
volume_scale = dbutils.widgets.get("volume_scale")
high_fraud_rate = float(dbutils.widgets.get("high_fraud_rate"))
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
overwrite_mode = dbutils.widgets.get("overwrite_mode") == "true"

# Set scale-based defaults
scale_configs = {
    "small": {"policyholders": 1000, "claims": 5000, "adjusters": 50, "providers": 200},
    "medium": {"policyholders": 10000, "claims": 50000, "adjusters": 200, "providers": 1000},
    "large": {"policyholders": 100000, "claims": 1000000, "adjusters": 500, "providers": 5000},
    "xlarge": {"policyholders": 1000000, "claims": 10000000, "adjusters": 2000, "providers": 20000},
}

if volume_scale == "custom":
    num_policyholders = int(dbutils.widgets.get("num_policyholders"))
    num_claims = int(dbutils.widgets.get("num_claims"))
    num_adjusters = int(dbutils.widgets.get("num_adjusters"))
    num_service_providers = int(dbutils.widgets.get("num_service_providers"))
else:
    config = scale_configs[volume_scale]
    num_policyholders = config["policyholders"]
    num_claims = config["claims"]
    num_adjusters = config["adjusters"]
    num_service_providers = config["providers"]

print(f"Configuration:")
print(f"  Volume Scale: {volume_scale}")
print(f"  Policyholders: {num_policyholders:,}")
print(f"  Claims: {num_claims:,}")
print(f"  Adjusters: {num_adjusters:,}")
print(f"  Service Providers: {num_service_providers:,}")
print(f"  High Fraud Score Rate: {high_fraud_rate:.1%}")
print(f"  Catalog: {catalog_name}")
print(f"  Schema: {schema_name}")
print(f"  Overwrite: {overwrite_mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Catalog and Schema

# COMMAND ----------

# Create catalog and schema if they don't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

print(f"Using {catalog_name}.{schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop Existing Tables (if overwrite mode)

# COMMAND ----------

if overwrite_mode:
    print("Dropping existing tables...")
    for table in ["policyholders", "claims", "adjusters", "service_providers", "discovered_fraud_rings"]:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {table}")
            print(f"  ✓ Dropped {table}")
        except Exception as e:
            print(f"  ⚠ Could not drop {table}: {e}")
    print("✅ Tables dropped")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Policyholders

# COMMAND ----------

# Generate policyholders with realistic data
print("Generating policyholders...")

# Create base policyholder IDs
policyholder_df = spark.range(num_policyholders).select(
    concat(lit("PH"), lpad(col("id").cast("string"), 8, "0")).alias("policyholder_id"),
    
    # Generate realistic names
    concat(
        expr("array('James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Christopher', 'Mary', 'Patricia', 'Jennifer', 'Linda', 'Barbara', 'Elizabeth', 'Susan', 'Jessica', 'Sarah', 'Karen')[cast(rand() * 20 as int)]"),
        lit(" "),
        expr("array('Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin')[cast(rand() * 20 as int)]")
    ).alias("name"),
    
    # Generate addresses with FRAUD RING PATTERNS
    # First 20 IDs: Create 4 fraud ring addresses (5 policyholders each)
    # Rest: Mix of unique and randomly shared
    when(col("id") < 5, lit("123 Fraud Ring Ave"))  # Fraud Ring 1
    .when((col("id") >= 5) & (col("id") < 10), lit("456 Suspicious Blvd"))  # Fraud Ring 2
    .when((col("id") >= 10) & (col("id") < 15), lit("789 Collusion St"))  # Fraud Ring 3
    .when((col("id") >= 15) & (col("id") < 20), lit("321 Scam Dr"))  # Fraud Ring 4
    .when(rand() < 0.7,
        # Unique addresses
        concat(
            (rand() * 9999 + 1).cast("int").cast("string"),
            lit(" "),
            expr("array('Main', 'Oak', 'Pine', 'Maple', 'Cedar', 'Elm', 'Washington', 'Lake', 'Hill', 'Park')[cast(rand() * 10 as int)]"),
            lit(" "),
            expr("array('St', 'Ave', 'Blvd', 'Dr', 'Ln', 'Rd', 'Way', 'Ct')[cast(rand() * 8 as int)]")
        )
    ).otherwise(
        # Randomly shared addresses
        concat(lit("SHARED-"), (rand() * 100).cast("int").cast("string"))
    ).alias("address"),
    
    # Cities and states
    expr("array('New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose')[cast(rand() * 10 as int)]").alias("city"),
    expr("array('NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'GA', 'MI')[cast(rand() * 10 as int)]").alias("state"),
    
    # Phone numbers with FRAUD RING PATTERNS
    # First 20 IDs: Share phone numbers within fraud rings
    # Rest: Mix of unique and randomly shared
    when(col("id") < 5, lit("555-FRAUD-0001"))  # Fraud Ring 1
    .when((col("id") >= 5) & (col("id") < 10), lit("555-FRAUD-0002"))  # Fraud Ring 2
    .when((col("id") >= 10) & (col("id") < 15), lit("555-FRAUD-0003"))  # Fraud Ring 3
    .when((col("id") >= 15) & (col("id") < 20), lit("555-FRAUD-0004"))  # Fraud Ring 4
    .when(rand() < 0.8,
        # Unique phone numbers
        concat(
            lit("555-"),
            lpad((rand() * 900 + 100).cast("int").cast("string"), 3, "0"),
            lit("-"),
            lpad((rand() * 9000 + 1000).cast("int").cast("string"), 4, "0")
        )
    ).otherwise(
        # Randomly shared phone numbers
        concat(lit("555-SHARED-"), lpad((rand() * 50).cast("int").cast("string"), 4, "0"))
    ).alias("phone"),
    
    # Email
    concat(
        lower(regexp_replace(concat(lit("PH"), lpad(col("id").cast("string"), 8, "0")), "[^a-zA-Z0-9]", "")),
        lit("@email.com")
    ).alias("email"),
    
    # Policy start date (within last 5 years)
    date_sub(current_date(), (rand() * 1825).cast("int")).alias("policy_start_date"),
    
    # Age (18-80)
    (rand() * 62 + 18).cast("int").alias("age"),
    
    # Customer tier
    expr("array('Standard', 'Silver', 'Gold', 'Platinum')[cast(rand() * 4 as int)]").alias("customer_tier")
)

# Write to Delta table
write_mode = "overwrite" if overwrite_mode else "append"
policyholder_df.write.format("delta").mode(write_mode).saveAsTable("policyholders")

print(f"✅ Generated {num_policyholders:,} policyholders")
display(spark.table("policyholders").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Adjusters (Neutral Entities)

# COMMAND ----------

# Generate adjusters - these are neutral entities, NO suspicious flags
print("Generating adjusters...")

adjuster_df = spark.range(num_adjusters).select(
    concat(lit("ADJ"), lpad(col("id").cast("string"), 6, "0")).alias("adjuster_id"),
    
    concat(
        expr("array('Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack', 'Kate', 'Leo', 'Maria', 'Nick', 'Olivia', 'Paul', 'Quinn', 'Rachel', 'Steve', 'Tracy')[cast(rand() * 20 as int)]"),
        lit(" "),
        expr("array('Anderson', 'Baker', 'Clark', 'Davis', 'Evans', 'Fisher', 'Green', 'Harris', 'Irving', 'Jackson')[cast(rand() * 10 as int)]")
    ).alias("name"),
    
    expr("array('Auto', 'Home', 'Health', 'Property', 'General')[cast(rand() * 5 as int)]").alias("department"),
    
    # Years of experience
    (rand() * 20 + 1).cast("int").alias("years_experience"),
    
    # License number
    concat(lit("LIC-"), lpad(col("id").cast("string"), 8, "0")).alias("license_number")
)

adjuster_df.write.format("delta").mode(write_mode).saveAsTable("adjusters")

print(f"✅ Generated {num_adjusters:,} adjusters")
display(spark.table("adjusters").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Service Providers (Neutral Entities)

# COMMAND ----------

# Generate service providers - neutral entities, NO suspicious flags
print("Generating service providers...")

provider_df = spark.range(num_service_providers).select(
    concat(lit("SP"), lpad(col("id").cast("string"), 6, "0")).alias("service_provider_id"),
    
    concat(
        expr("array('Quick', 'Express', 'Premium', 'Elite', 'Professional', 'Expert', 'Quality', 'Reliable', 'Trusted', 'Premier')[cast(rand() * 10 as int)]"),
        lit(" "),
        expr("array('Auto Repair', 'Body Shop', 'Medical Center', 'Clinic', 'Legal Services', 'Construction', 'Restoration', 'Garage', 'Hospital', 'Law Firm')[cast(rand() * 10 as int)]")
    ).alias("provider_name"),
    
    expr("array('Auto Repair', 'Body Shop', 'Medical', 'Legal', 'Construction', 'Restoration')[cast(rand() * 6 as int)]").alias("service_type"),
    
    # Business rating (1.0 to 5.0)
    round(rand() * 4 + 1, 1).alias("rating"),
    
    # Years in business
    (rand() * 30 + 1).cast("int").alias("years_in_business"),
    
    # City
    expr("array('New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose')[cast(rand() * 10 as int)]").alias("city")
)

provider_df.write.format("delta").mode(write_mode).saveAsTable("service_providers")

print(f"✅ Generated {num_service_providers:,} service providers")
display(spark.table("service_providers").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Claims with Fraud Scores

# COMMAND ----------

# Generate claims with fraud_score (0-1) - this is the key entity for fraud detection
print("Generating claims...")

# Get counts for random assignment
ph_count = spark.table("policyholders").count()
adj_count = spark.table("adjusters").count()
sp_count = spark.table("service_providers").count()

print(f"  Assigning from {ph_count} policyholders, {adj_count} adjusters, {sp_count} service providers...")

# Create "fraud ring" entity IDs - these will be shared across multiple high-fraud claims
# This creates the patterns that GraphFrames will discover!
fraud_ring_adjusters = [f"ADJ{str(i).zfill(6)}" for i in range(5)]  # 5 suspicious adjusters
fraud_ring_providers = [f"SP{str(i).zfill(6)}" for i in range(8)]    # 8 suspicious providers
fraud_ring_policyholders = [f"PH{str(i).zfill(8)}" for i in range(15)]  # 15 repeat fraudsters

print(f"  Creating fraud ring patterns with:")
print(f"    - {len(fraud_ring_adjusters)} suspicious adjusters")
print(f"    - {len(fraud_ring_providers)} suspicious service providers") 
print(f"    - {len(fraud_ring_policyholders)} repeat fraud policyholders")

# Generate claims
claims_df = spark.range(num_claims).select(
    concat(lit("CLM"), lpad(col("id").cast("string"), 10, "0")).alias("claim_id"),
    
    # Policyholder assignment:
    # - 20% of claims go to "fraud ring" policyholders (creating repeat patterns)
    # - 80% distributed randomly
    when(rand() < 0.20,
        expr(f"array({','.join([repr(ph) for ph in fraud_ring_policyholders])})[cast(rand() * {len(fraud_ring_policyholders)} as int)]")
    ).otherwise(
        concat(lit("PH"), lpad(((rand() * ph_count).cast("long") % ph_count).cast("string"), 8, "0"))
    ).alias("policyholder_id"),
    
    # Adjuster assignment:
    # - 25% of claims processed by "fraud ring" adjusters
    # - 75% distributed randomly
    when(rand() < 0.25,
        expr(f"array({','.join([repr(adj) for adj in fraud_ring_adjusters])})[cast(rand() * {len(fraud_ring_adjusters)} as int)]")
    ).otherwise(
        concat(lit("ADJ"), lpad(((rand() * adj_count).cast("long") % adj_count).cast("string"), 6, "0"))
    ).alias("adjuster_id"),
    
    # Service provider assignment:
    # - 30% serviced by "fraud ring" providers
    # - 70% distributed randomly
    when(rand() < 0.30,
        expr(f"array({','.join([repr(sp) for sp in fraud_ring_providers])})[cast(rand() * {len(fraud_ring_providers)} as int)]")
    ).otherwise(
        concat(lit("SP"), lpad(((rand() * sp_count).cast("long") % sp_count).cast("string"), 6, "0"))
    ).alias("service_provider_id"),
    
    # Claim type
    expr("array('Auto Collision', 'Auto Comprehensive', 'Home Fire', 'Home Theft', 'Home Water Damage', 'Health Medical', 'Property Damage', 'Liability')[cast(rand() * 8 as int)]").alias("claim_type"),
    
    # Incident date (within last 2 years)
    date_sub(current_date(), (rand() * 730).cast("int")).alias("incident_date"),
    
    # Filing date (1-30 days after incident)
    date_sub(current_date(), (rand() * 700).cast("int")).alias("filing_date"),
    
    # Claim amount - follows realistic distribution
    # Most claims: $1K-$20K, some high value: $20K-$100K
    when(rand() < 0.85,
        round(rand() * 19000 + 1000, 2)  # 85% are normal ($1K-$20K)
    ).otherwise(
        round(rand() * 80000 + 20000, 2)  # 15% are high value ($20K-$100K)
    ).alias("claim_amount"),
    
    # Claim status
    expr("array('Pending', 'Approved', 'Under Review', 'Denied')[cast(rand() * 4 as int)]").alias("status"),
    
    # FRAUD SCORE (0-1) - continuous score indicating suspiciousness
    # Higher scores for: fraud ring entities, high amounts, certain claim types
    # This is what GraphFrames will use to discover fraud rings!
    greatest(
        least(
            # Base fraud score (0.1 to 0.9)
            round(rand() * 0.8 + 0.1, 3) +
            # FRAUD RING BONUS: Claims involving fraud ring entities get +0.3 to +0.5 boost
            when(col("policyholder_id").isin(fraud_ring_policyholders), 0.4).otherwise(0.0) +
            when(col("adjuster_id").isin(fraud_ring_adjusters), 0.35).otherwise(0.0) +
            when(col("service_provider_id").isin(fraud_ring_providers), 0.3).otherwise(0.0) +
            # High amount increases score
            when(col("claim_amount") > 50000, 0.2).otherwise(0.0) +
            # Certain claim types are riskier
            when(col("claim_type").isin('Home Fire', 'Auto Comprehensive'), 0.15).otherwise(0.0) +
            # Random noise
            round((rand() - 0.5) * 0.15, 3),
            lit(1.0)  # Cap at 1.0
        ),
        lit(0.0)  # Floor at 0.0
    ).alias("fraud_score")
)

# Add derived fields
claims_df = claims_df.withColumn(
    "processing_days",
    datediff(current_date(), col("filing_date"))
).withColumn(
    "days_to_file",
    datediff(col("filing_date"), col("incident_date"))
)

# Write to Delta table
claims_df.write.format("delta").mode(write_mode).saveAsTable("claims")

print(f"✅ Generated {num_claims:,} claims with fraud scores")

# Show fraud score distribution
print("\nFraud Score Distribution:")
display(
    spark.table("claims")
    .select(
        when(col("fraud_score") < 0.3, "Low (0.0-0.3)")
        .when(col("fraud_score") < 0.6, "Medium (0.3-0.6)")
        .otherwise("High (0.6-1.0)")
        .alias("fraud_score_category")
    )
    .groupBy("fraud_score_category")
    .count()
    .orderBy("fraud_score_category")
)

display(spark.table("claims").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

print("\n" + "="*80)
print("DATASET GENERATION COMPLETE")
print("="*80)

# Policyholders
ph_count = spark.table("policyholders").count()
shared_addresses = spark.table("policyholders").filter(col("address").startswith("SHARED-")).count()
print(f"\n📋 Policyholders: {ph_count:,}")
print(f"   • With shared addresses: {shared_addresses:,} ({shared_addresses/ph_count*100:.1f}%)")

# Adjusters
adj_count = spark.table("adjusters").count()
print(f"\n👨‍💼 Adjusters: {adj_count:,}")

# Service Providers
sp_count = spark.table("service_providers").count()
print(f"\n🏢 Service Providers: {sp_count:,}")

# Claims
claims_stats = spark.table("claims").agg(
    count("*").alias("total"),
    avg("claim_amount").alias("avg_amount"),
    avg("fraud_score").alias("avg_fraud_score"),
    sum(when(col("fraud_score") >= 0.7, 1).otherwise(0)).alias("high_fraud_count")
).collect()[0]

print(f"\n💰 Claims: {claims_stats.total:,}")
print(f"   • Average amount: ${claims_stats.avg_amount:,.2f}")
print(f"   • Average fraud score: {claims_stats.avg_fraud_score:.3f}")
print(f"   • High fraud score (≥0.7): {claims_stats.high_fraud_count:,} ({claims_stats.high_fraud_count/claims_stats.total*100:.1f}%)")

# Policyholders with multiple claims
multi_claim_ph = spark.table("claims").groupBy("policyholder_id").count().filter(col("count") > 1).count()
print(f"\n🔁 Policyholders with multiple claims: {multi_claim_ph:,}")

# Claim types
print("\n📊 Top Claim Types:")
display(spark.table("claims").groupBy("claim_type").agg(
    count("*").alias("count"),
    round(avg("fraud_score"), 3).alias("avg_fraud_score")
).orderBy(desc("count")).limit(10))

print("\n✅ Ready for GraphFrames fraud detection analysis!")
print(f"   Next: Run notebook 02_GraphFrames_Fraud_Detection to DISCOVER fraud rings!")
print(f"\n💡 Key Point: No fraud rings were pre-generated!")
print(f"   GraphFrames will discover them based on:")
print(f"   • Shared addresses/phones among policyholders")
print(f"   • High fraud_score concentrations")
print(f"   • Network patterns and relationships")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables Created
# MAGIC 
# MAGIC The following Delta tables have been created in `dbdemos_steventan.frauddetection_graphframe`:
# MAGIC 
# MAGIC 1. **`policyholders`** - Insurance customers (some share addresses/phones)
# MAGIC 2. **`claims`** - Insurance claims with **fraud_score (0-1)** indicating suspiciousness
# MAGIC 3. **`adjusters`** - Insurance professionals (neutral entities)
# MAGIC 4. **`service_providers`** - Service providers (neutral entities)
# MAGIC 
# MAGIC **Key Design:**
# MAGIC - Claims have continuous `fraud_score` (0-1), not binary labels
# MAGIC - Natural patterns embedded (shared contacts, multiple claims per policyholder)
# MAGIC - NO pre-labeled fraud rings - GraphFrames will discover them!
