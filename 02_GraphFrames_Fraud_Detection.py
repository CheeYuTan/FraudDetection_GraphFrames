# Databricks notebook source
# MAGIC %md
# MAGIC # Insurance Fraud Detection with GraphFrames
# MAGIC 
# MAGIC This notebook demonstrates how to **DISCOVER** fraud rings using **GraphFrames** algorithms.
# MAGIC 
# MAGIC ## 🔍 What We'll Discover:
# MAGIC 
# MAGIC Starting with only:
# MAGIC - Claims with **fraud_score** (0-1)
# MAGIC - Relationships between claims, policyholders, adjusters, service providers
# MAGIC 
# MAGIC We'll use GraphFrames to **discover**:
# MAGIC - **Fraud rings**: Groups of connected entities with high fraud scores
# MAGIC - **Influential nodes**: Key players in suspicious networks  
# MAGIC - **Suspicious patterns**: Shared contacts, coordinated claims
# MAGIC - **Risk clusters**: Communities with concentrated fraud
# MAGIC 
# MAGIC ## 🚀 GraphFrames Algorithms Used:
# MAGIC 
# MAGIC 1. **Connected Components** - Find all connected groups
# MAGIC 2. **PageRank** - Identify influential entities in high-fraud networks
# MAGIC 3. **Label Propagation** - Detect fraud communities
# MAGIC 4. **Triangle Count** - Measure clustering/collusion
# MAGIC 5. **Motif Finding** - Discover suspicious relationship patterns
# MAGIC 6. **Shortest Paths** - Analyze proximity to high-fraud entities
# MAGIC 7. **BFS** - Investigate networks from suspicious claims
# MAGIC 
# MAGIC ## 💡 Key Innovation:
# MAGIC **We DON'T know which networks are fraud rings upfront!**  
# MAGIC GraphFrames discovers them by analyzing fraud_score patterns in the network structure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Libraries
# MAGIC 
# MAGIC Installing GraphFrames and visualization libraries for serverless compute.

# COMMAND ----------

# Install required libraries (serverless compatible)
%pip install graphframes networkx matplotlib pandas --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Create widgets
dbutils.widgets.text("catalog_name", "dbdemos_steventan", "Catalog Name")
dbutils.widgets.text("schema_name", "frauddetection_graphframe", "Schema Name")
dbutils.widgets.text("fraud_threshold", "0.7", "High Fraud Score Threshold")
dbutils.widgets.text("min_network_size", "3", "Minimum Network Size for Analysis")

# Get configuration
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
fraud_threshold = float(dbutils.widgets.get("fraud_threshold"))
min_network_size = int(dbutils.widgets.get("min_network_size"))

# Use catalog and schema
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

print(f"Using {catalog_name}.{schema_name}")
print(f"Fraud threshold: {fraud_threshold}")
print(f"Minimum network size: {min_network_size}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from graphframes import GraphFrame
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load tables
policyholders = spark.table("policyholders")
claims = spark.table("claims")
adjusters = spark.table("adjusters")
service_providers = spark.table("service_providers")

# Show statistics
print("Data loaded:")
print(f"  Policyholders: {policyholders.count():,}")
print(f"  Claims: {claims.count():,}")
print(f"  Adjusters: {adjusters.count():,}")
print(f"  Service Providers: {service_providers.count():,}")

# Fraud score distribution
fraud_stats = claims.agg(
    avg("fraud_score").alias("avg_score"),
    min("fraud_score").alias("min_score"),
    max("fraud_score").alias("max_score"),
    sum(when(col("fraud_score") >= fraud_threshold, 1).otherwise(0)).alias("high_fraud_count")
).collect()[0]

print(f"\nFraud Score Statistics:")
print(f"  Average: {fraud_stats.avg_score:.3f}")
print(f"  Range: {fraud_stats.min_score:.3f} - {fraud_stats.max_score:.3f}")
print(f"  High fraud scores (≥{fraud_threshold}): {fraud_stats.high_fraud_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 📊 Build Graph Structure

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vertices (Nodes)

# COMMAND ----------

# Create vertices from all entities
# Claims as vertices (with fraud_score)
claim_vertices = claims.select(
    col("claim_id").alias("id"),
    lit("claim").alias("type"),
    col("fraud_score"),
    col("claim_amount"),
    col("claim_type"),
    col("status"),
    col("incident_date")
)

# Policyholders as vertices
policyholder_vertices = policyholders.select(
    col("policyholder_id").alias("id"),
    lit("policyholder").alias("type"),
    lit(None).cast("double").alias("fraud_score"),
    lit(None).cast("decimal(10,2)").alias("claim_amount"),
    col("address").alias("claim_type"),  # Store address for analysis
    col("phone").alias("status"),  # Store phone for analysis
    col("policy_start_date").alias("incident_date")
)

# Adjusters as vertices (neutral)
adjuster_vertices = adjusters.select(
    col("adjuster_id").alias("id"),
    lit("adjuster").alias("type"),
    lit(None).cast("double").alias("fraud_score"),
    lit(None).cast("decimal(10,2)").alias("claim_amount"),
    col("department").alias("claim_type"),
    col("name").alias("status"),
    lit(None).cast("date").alias("incident_date")
)

# Service providers as vertices (neutral)
provider_vertices = service_providers.select(
    col("service_provider_id").alias("id"),
    lit("provider").alias("type"),
    lit(None).cast("double").alias("fraud_score"),
    lit(None).cast("decimal(10,2)").alias("claim_amount"),
    col("service_type").alias("claim_type"),
    col("provider_name").alias("status"),
    lit(None).cast("date").alias("incident_date")
)

# Union all vertices
vertices = claim_vertices.union(policyholder_vertices).union(adjuster_vertices).union(provider_vertices)

print(f"Total vertices: {vertices.count():,}")
display(vertices.groupBy("type").count().orderBy(desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Edges (Relationships)

# COMMAND ----------

# Create edges representing relationships
# 1. Claim -> Policyholder (filed_by) - include claim's fraud_score
claim_policyholder_edges = claims.select(
    col("claim_id").alias("src"),
    col("policyholder_id").alias("dst"),
    lit("filed_by").alias("relationship"),
    col("filing_date").alias("edge_date"),
    col("fraud_score")
)

# 2. Claim -> Adjuster (processed_by)
claim_adjuster_edges = claims.select(
    col("claim_id").alias("src"),
    col("adjuster_id").alias("dst"),
    lit("processed_by").alias("relationship"),
    col("filing_date").alias("edge_date"),
    col("fraud_score")
)

# 3. Claim -> Service Provider (serviced_by)
claim_provider_edges = claims.select(
    col("claim_id").alias("src"),
    col("service_provider_id").alias("dst"),
    lit("serviced_by").alias("relationship"),
    col("incident_date").alias("edge_date"),
    col("fraud_score")
)

# 4. Policyholder -> Policyholder (shared_contact)
# This is KEY for discovering fraud rings!
ph_shared = policyholders.alias("p1").join(
    policyholders.alias("p2"),
    (
        (col("p1.policyholder_id") < col("p2.policyholder_id")) &
        ((col("p1.address") == col("p2.address")) | (col("p1.phone") == col("p2.phone")))
    )
).select(
    col("p1.policyholder_id").alias("src"),
    col("p2.policyholder_id").alias("dst"),
    lit("shared_contact").alias("relationship"),
    current_date().alias("edge_date"),
    lit(None).cast("double").alias("fraud_score")
)

# 5. Policyholder -> Adjuster (repeat collaboration - CREATES TRIANGLES!)
# When a policyholder files multiple claims with the same adjuster
ph_adj_pairs = claims.groupBy("policyholder_id", "adjuster_id").agg(
    count("*").alias("claim_count"),
    avg("fraud_score").alias("avg_fraud_score")
).filter(col("claim_count") >= 2)  # At least 2 claims together

ph_adjuster_edges = ph_adj_pairs.select(
    col("policyholder_id").alias("src"),
    col("adjuster_id").alias("dst"),
    lit("repeat_adjuster").alias("relationship"),
    current_date().alias("edge_date"),
    col("avg_fraud_score").alias("fraud_score")
)

# 6. Policyholder -> Provider (repeat collaboration - CREATES TRIANGLES!)
# When a policyholder uses the same service provider multiple times
ph_prov_pairs = claims.groupBy("policyholder_id", "service_provider_id").agg(
    count("*").alias("claim_count"),
    avg("fraud_score").alias("avg_fraud_score")
).filter(col("claim_count") >= 2)  # At least 2 claims together

ph_provider_edges = ph_prov_pairs.select(
    col("policyholder_id").alias("src"),
    col("service_provider_id").alias("dst"),
    lit("repeat_provider").alias("relationship"),
    current_date().alias("edge_date"),
    col("avg_fraud_score").alias("fraud_score")
)

# 7. Adjuster -> Provider (frequent collaboration - CREATES TRIANGLES!)
# When an adjuster and provider appear together on multiple claims
adj_prov_pairs = claims.groupBy("adjuster_id", "service_provider_id").agg(
    count("*").alias("claim_count"),
    avg("fraud_score").alias("avg_fraud_score")
).filter(col("claim_count") >= 3)  # At least 3 claims together (stronger signal)

adjuster_provider_edges = adj_prov_pairs.select(
    col("adjuster_id").alias("src"),
    col("service_provider_id").alias("dst"),
    lit("frequent_collaboration").alias("relationship"),
    current_date().alias("edge_date"),
    col("avg_fraud_score").alias("fraud_score")
)

# Union all edges - NOW WITH TRIANGLE-FORMING EDGES!
edges = claim_policyholder_edges.union(claim_adjuster_edges).union(claim_provider_edges).union(ph_shared).union(ph_adjuster_edges).union(ph_provider_edges).union(adjuster_provider_edges)

print(f"Total edges: {edges.count():,}")
display(edges.groupBy("relationship").count().orderBy(desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create GraphFrame

# COMMAND ----------

# Create the graph (serverless compatible - no caching)
print("Creating GraphFrame...")
g = GraphFrame(vertices, edges)

print("✅ Graph created successfully!")
print(f"  Vertices: {g.vertices.count():,}")
print(f"  Edges: {g.edges.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Initial Graph Visualization
# MAGIC 
# MAGIC Let's visualize a sample of the graph to understand its structure

# COMMAND ----------

# Visualize a sample of the graph
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

print("Creating graph visualization...")

# Better approach: Sample HIGH-FRAUD claims and their connections
print("  Sampling high-fraud claims and their networks...")

# Get high-fraud claims (fraud_score >= 0.7)
high_fraud_claims = g.vertices.filter(
    (col("type") == "claim") & (col("fraud_score") >= 0.7)
).limit(5).select("id").toPandas()['id'].tolist()

if len(high_fraud_claims) > 0:
    # Get edges involving these high-fraud claims
    fraud_edges = g.edges.filter(
        col("src").isin(high_fraud_claims) | col("dst").isin(high_fraud_claims)
    ).toPandas()
    
    # Also get some shared_contact edges (fraud indicators!)
    shared_edges = g.edges.filter(col("relationship") == "shared_contact").limit(5).toPandas()
    
    # Combine
    sample_edges = pd.concat([fraud_edges, shared_edges]).drop_duplicates()
    
    # Get all vertices involved
    edge_vertices = list(set(sample_edges['src'].tolist() + sample_edges['dst'].tolist()))
    sample_vertices = g.vertices.filter(col("id").isin(edge_vertices)).toPandas()
else:
    # Fallback: diverse sampling
    filed_by_sample = g.edges.filter(col("relationship") == "filed_by").limit(8)
    processed_by_sample = g.edges.filter(col("relationship") == "processed_by").limit(8)
    serviced_by_sample = g.edges.filter(col("relationship") == "serviced_by").limit(8)
    shared_contact_sample = g.edges.filter(col("relationship") == "shared_contact").limit(6)
    
    sample_edges = filed_by_sample.union(processed_by_sample).union(serviced_by_sample).union(shared_contact_sample).toPandas()
    edge_vertices = list(set(sample_edges['src'].tolist() + sample_edges['dst'].tolist()))
    sample_vertices = g.vertices.filter(col("id").isin(edge_vertices)).toPandas()

print(f"  Sampled {len(sample_vertices)} vertices and {len(sample_edges)} edges")
print(f"  Focusing on high-fraud claims (score >= 0.7) and their connections")

# Create NetworkX graph
G = nx.DiGraph()

# Add nodes with attributes
for _, row in sample_vertices.iterrows():
    G.add_node(row['id'], 
              node_type=row['type'],
              fraud_score=row['fraud_score'] if pd.notna(row['fraud_score']) else 0)

# Add edges - make sure both nodes exist
for _, row in sample_edges.iterrows():
    if row['src'] in G.nodes() and row['dst'] in G.nodes():
        G.add_edge(row['src'], row['dst'], 
                  relationship=row['relationship'],
                  fraud_score=row['fraud_score'] if pd.notna(row['fraud_score']) else 0)

print(f"  Graph built: {len(G.nodes())} nodes, {len(G.edges())} edges")

# Create visualization
fig, ax = plt.subplots(figsize=(20, 16))

# Layout with good spacing
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Separate nodes by type (use different variable names to avoid overwriting DataFrames!)
claim_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'claim']
policyholder_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'policyholder']
adjuster_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'adjuster']
provider_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'provider']

# Color claims by fraud score
claim_colors = [G.nodes[n].get('fraud_score', 0) for n in claim_nodes]

# Draw nodes (moderate size)
nx.draw_networkx_nodes(G, pos, nodelist=claim_nodes, 
                      node_color=claim_colors, cmap='RdYlGn_r',
                      node_size=600, node_shape='o', label='Claims', alpha=0.8,
                      vmin=0, vmax=1, edgecolors='black', linewidths=1.5)
nx.draw_networkx_nodes(G, pos, nodelist=policyholder_nodes, node_color='orange', 
                      node_size=500, node_shape='s', label='Policyholders', alpha=0.8,
                      edgecolors='black', linewidths=1.5)
nx.draw_networkx_nodes(G, pos, nodelist=adjuster_nodes, node_color='lightgreen', 
                      node_size=500, node_shape='^', label='Adjusters', alpha=0.8,
                      edgecolors='black', linewidths=1.5)
nx.draw_networkx_nodes(G, pos, nodelist=provider_nodes, node_color='mediumpurple', 
                      node_size=500, node_shape='v', label='Providers', alpha=0.8,
                      edgecolors='black', linewidths=1.5)

# Draw edges with different colors by relationship type
filed_by_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'filed_by']
processed_by_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'processed_by']
serviced_by_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'serviced_by']
shared_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'shared_contact']

# Draw different edge types with different colors (smaller edges)
nx.draw_networkx_edges(G, pos, edgelist=filed_by_edges, edge_color='blue', 
                      arrows=True, arrowsize=10, alpha=0.6, width=1.5, label='Filed by')
nx.draw_networkx_edges(G, pos, edgelist=processed_by_edges, edge_color='green', 
                      arrows=True, arrowsize=10, alpha=0.5, width=1.2, style='dashed', label='Processed by')
nx.draw_networkx_edges(G, pos, edgelist=serviced_by_edges, edge_color='purple', 
                      arrows=True, arrowsize=10, alpha=0.5, width=1.2, style='dotted', label='Serviced by')
nx.draw_networkx_edges(G, pos, edgelist=shared_edges, edge_color='red', 
                      arrows=False, alpha=0.7, width=2, label='Shared contact')

# Add labels (smaller font, abbreviated IDs)
labels = {node: node[:8] + '...' if len(node) > 8 else node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold')

# Color bar for fraud scores
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Fraud Score', rotation=270, labelpad=20)

plt.title(f"Insurance Claims Network Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)\n" +
         "Claims colored by fraud score | Blue edges = Filed by | Green = Processed by | Purple = Serviced by | Red = Shared contact", 
         fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10, ncol=2)
plt.axis('off')
plt.tight_layout()

print(f"\n📊 Visualization shows {len(G.nodes())} nodes and {len(G.edges())} edges")
print(f"   • Claims: {len(claim_nodes)}")
print(f"   • Policyholders: {len(policyholder_nodes)}")
print(f"   • Adjusters: {len(adjuster_nodes)}")
print(f"   • Providers: {len(provider_nodes)}")
print(f"\n   Edge types:")
print(f"   • Filed by (blue): {len(filed_by_edges)}")
print(f"   • Processed by (green dashed): {len(processed_by_edges)}")
print(f"   • Serviced by (purple dotted): {len(serviced_by_edges)}")
print(f"   • Shared contact (red thick): {len(shared_edges)}")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 🔍 DISCOVER FRAUD RINGS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Find All Connected Components
# MAGIC 
# MAGIC First, we identify all connected groups in the network.  
# MAGIC We DON'T know which ones are fraud rings yet!

# COMMAND ----------

# Set checkpoint directory (required for GraphFrames algorithms)
print("Setting up checkpoint directory...")
import tempfile
import os

# Create temporary checkpoint directory
checkpoint_dir = f"/tmp/graphframes_checkpoints/{os.getpid()}"
spark.sparkContext.setCheckpointDir(checkpoint_dir)
print(f"✅ Checkpoint directory set to: {checkpoint_dir}")

# COMMAND ----------

print("Running Connected Components...")
cc_results = g.connectedComponents()

# Analyze each component
components = cc_results.select("id", "type", "fraud_score", "component")

# Calculate statistics for each component
component_stats = components.groupBy("component").agg(
    count("*").alias("total_entities"),
    sum(when(col("type") == "claim", 1).otherwise(0)).alias("claim_count"),
    sum(when(col("type") == "policyholder", 1).otherwise(0)).alias("policyholder_count"),
    sum(when(col("type") == "adjuster", 1).otherwise(0)).alias("adjuster_count"),
    sum(when(col("type") == "provider", 1).otherwise(0)).alias("provider_count"),
    
    # KEY METRICS: Average fraud score in this network
    round(avg(when(col("type") == "claim", col("fraud_score"))), 3).alias("avg_fraud_score"),
    round(max(when(col("type") == "claim", col("fraud_score"))), 3).alias("max_fraud_score"),
    
    # Count high fraud claims
    sum(when((col("type") == "claim") & (col("fraud_score") >= fraud_threshold), 1).otherwise(0)).alias("high_fraud_claims")
).withColumn(
    "fraud_concentration",
    when(col("claim_count") > 0, 
         round(col("high_fraud_claims") / col("claim_count") * 100, 1)
    ).otherwise(0.0)
).orderBy(desc("claim_count"))

print(f"\n📊 Found {component_stats.count():,} connected components")
print("\nTop 20 components by size:")
display(component_stats.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Visualization: Connected Components
# MAGIC 
# MAGIC Visualize the largest connected components colored by component ID

# COMMAND ----------

# Visualize connected components - focus on high-fraud components
print("Creating connected components visualization...")

# Get components with high fraud scores (skip the giant component)
interesting_components = component_stats.filter(
    (col("claim_count") >= 3) & (col("claim_count") <= 100) &  # Medium-sized components
    (col("avg_fraud_score") >= 0.5)  # High fraud score
).orderBy(desc("avg_fraud_score")).limit(3).select("component").toPandas()

if len(interesting_components) > 0:
    top_components = interesting_components['component'].tolist()
    
    # Get ALL vertices in these interesting components (they're small enough)
    viz_vertices = components.filter(col("component").isin(top_components)).toPandas()
    viz_vertex_ids = viz_vertices['id'].tolist()
    
    viz_edges = g.edges.filter(
        col("src").isin(viz_vertex_ids) & col("dst").isin(viz_vertex_ids)
    ).toPandas()
else:
    # Fallback: sample high-fraud claims and their immediate connections
    print("  No interesting components found, sampling high-fraud claims instead...")
    high_fraud_claims = components.filter(
        (col("type") == "claim") & (col("fraud_score") >= 0.7)
    ).limit(10).select("id").toPandas()['id'].tolist()
    
    # Get edges connected to these high-fraud claims
    viz_edges = g.edges.filter(
        col("src").isin(high_fraud_claims) | col("dst").isin(high_fraud_claims)
    ).limit(40).toPandas()
    
    # Get all vertices in these edges
    edge_verts = list(set(viz_edges['src'].tolist() + viz_edges['dst'].tolist()))
    viz_vertices = components.filter(col("id").isin(edge_verts)).toPandas()
    viz_vertex_ids = viz_vertices['id'].tolist()
    top_components = viz_vertices['component'].unique().tolist()

# Create NetworkX graph
G_comp = nx.DiGraph()

for _, row in viz_vertices.iterrows():
    G_comp.add_node(row['id'], 
                   component=row['component'],
                   node_type=row['type'],
                   fraud_score=row['fraud_score'] if pd.notna(row['fraud_score']) else 0)

for _, row in viz_edges.iterrows():
    G_comp.add_edge(row['src'], row['dst'])

# Visualization
fig, ax = plt.subplots(figsize=(20, 16))
pos = nx.spring_layout(G_comp, k=3, iterations=50, seed=42)

# Color by component
component_colors = [G_comp.nodes[n]['component'] for n in G_comp.nodes()]
unique_components = list(set(component_colors))
color_map = {comp: i for i, comp in enumerate(unique_components)}
node_colors = [color_map[G_comp.nodes[n]['component']] for n in G_comp.nodes()]

# Node sizes by fraud score
node_sizes = [G_comp.nodes[n].get('fraud_score', 0.5) * 1000 + 300 for n in G_comp.nodes()]

nx.draw_networkx_nodes(G_comp, pos, node_color=node_colors, node_size=node_sizes,
                      cmap='tab10', alpha=0.8)
nx.draw_networkx_edges(G_comp, pos, edge_color='gray', arrows=True, 
                      arrowsize=10, alpha=0.3, width=1.5)

labels = {node: node[:8] if len(node) > 8 else node for node in G_comp.nodes()}
nx.draw_networkx_labels(G_comp, pos, labels, font_size=6)

plt.title(f"Connected Components Visualization (Top {len(unique_components)} Components)\n" +
         "Each color = Different component | Node size = Fraud score", 
         fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"\n📊 Showing {len(G_comp.nodes())} nodes across {len(unique_components)} components")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Identify Suspected Fraud Rings
# MAGIC 
# MAGIC **Fraud Ring Criteria:**
# MAGIC - Contains multiple claims (≥3)
# MAGIC - High average fraud score (≥0.6)
# MAGIC - OR high fraud concentration (≥40% of claims have score ≥0.7)

# COMMAND ----------

# Identify suspected fraud rings based on fraud score patterns
suspected_fraud_rings = component_stats.filter(
    (col("claim_count") >= min_network_size) &
    ((col("avg_fraud_score") >= 0.6) | (col("fraud_concentration") >= 40))
).orderBy(desc("fraud_concentration"), desc("avg_fraud_score"))

print(f"\n🚨 DISCOVERED {suspected_fraud_rings.count():,} SUSPECTED FRAUD RINGS!")
print("\nSuspected Fraud Rings (sorted by fraud concentration):")
display(suspected_fraud_rings)

# Save for later analysis
suspected_fraud_rings.write.format("delta").mode("overwrite").saveAsTable("discovered_fraud_rings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Analyze a Sample Fraud Ring

# COMMAND ----------

if suspected_fraud_rings.count() > 0:
    # Get the most suspicious fraud ring
    top_fraud_ring = suspected_fraud_rings.first()
    ring_component_id = top_fraud_ring["component"]
    
    print(f"\n🔍 Analyzing Top Fraud Ring (Component {ring_component_id}):")
    print(f"   Total entities: {top_fraud_ring.total_entities}")
    print(f"   Claims: {top_fraud_ring.claim_count}")
    print(f"   Policyholders: {top_fraud_ring.policyholder_count}")
    print(f"   Avg fraud score: {top_fraud_ring.avg_fraud_score}")
    print(f"   Fraud concentration: {top_fraud_ring.fraud_concentration}%")
    
    # Get all entities in this fraud ring
    fraud_ring_entities = components.filter(col("component") == ring_component_id)
    
    print("\n📋 Entities in this fraud ring:")
    display(fraud_ring_entities.groupBy("type").count())
    
    # Show claims in this ring
    print("\n💰 Claims in this fraud ring:")
    # Use aliases to avoid ambiguous column references
    fraud_ring_claim_ids = fraud_ring_entities.filter(col("type") == "claim").select(col("id").alias("ring_id"))
    fraud_ring_claims = claims.alias("c").join(
        fraud_ring_claim_ids.alias("ring"), 
        col("c.claim_id") == col("ring.ring_id")
    ).select(
        col("c.claim_id"), 
        col("c.policyholder_id"), 
        col("c.claim_amount"), 
        col("c.fraud_score"), 
        col("c.claim_type"), 
        col("c.incident_date")
    ).orderBy(desc(col("c.fraud_score")))
    
    display(fraud_ring_claims)
    
    # Show policyholders in this ring
    print("\n👥 Policyholders in this fraud ring:")
    # Use aliases to avoid ambiguous column references
    fraud_ring_ph_ids = fraud_ring_entities.filter(col("type") == "policyholder").select(col("id").alias("ring_id"))
    fraud_ring_policyholders = policyholders.alias("ph").join(
        fraud_ring_ph_ids.alias("ring"), 
        col("ph.policyholder_id") == col("ring.ring_id")
    ).select(
        col("ph.policyholder_id"), 
        col("ph.name"), 
        col("ph.address"), 
        col("ph.phone"), 
        col("ph.city"), 
        col("ph.state")
    )
    
    display(fraud_ring_policyholders)
    
    # Check for shared contacts
    shared_addresses = fraud_ring_policyholders.groupBy("address").count().filter(col("count") > 1)
    shared_phones = fraud_ring_policyholders.groupBy("phone").count().filter(col("count") > 1)
    
    print(f"\n🔗 Shared contact patterns:")
    print(f"   Shared addresses: {shared_addresses.count()}")
    print(f"   Shared phones: {shared_phones.count()}")
    
    if shared_addresses.count() > 0:
        print("\n   Addresses shared by multiple policyholders:")
        display(shared_addresses)
else:
    print("No suspected fraud rings found with current criteria")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: PageRank on High-Fraud Networks
# MAGIC 
# MAGIC Within suspected fraud rings, find the most influential entities (potential ringleaders)

# COMMAND ----------

if suspected_fraud_rings.count() > 0:
    print("Running PageRank on the full graph...")
    pagerank_results = g.pageRank(resetProbability=0.15, maxIter=10)
    
    # Get top entities by PageRank
    top_pagerank = pagerank_results.vertices.select(
        "id", "type", "fraud_score", "pagerank"
    ).orderBy(desc("pagerank"))
    
    print("\n🏆 Top 20 Most Influential Entities (by PageRank):")
    display(top_pagerank.limit(20))
    
    # Focus on influential entities IN suspected fraud rings
    fraud_ring_components = suspected_fraud_rings.select("component")
    
    # Use aliases to avoid ambiguous column references
    influential_in_fraud_rings = pagerank_results.vertices.alias("pr").join(
        components.alias("comp"), col("pr.id") == col("comp.id")
    ).join(
        fraud_ring_components.alias("frc"), col("comp.component") == col("frc.component")
    ).filter(
        col("pr.type").isin("claim", "policyholder")
    ).select(
        col("pr.id"), 
        col("pr.type"), 
        col("pr.fraud_score"), 
        col("pr.pagerank"), 
        col("comp.component")
    ).orderBy(desc(col("pr.pagerank")))
    
    print("\n🚨 Most Influential Entities in Suspected Fraud Rings (Potential Ringleaders):")
    display(influential_in_fraud_rings.limit(20))
    
    # Visualize PageRank results
    print("\n" + "="*60)
    print("📊 VISUALIZATION: PageRank Results")
    print("="*60)
    print("Creating PageRank visualization...")
    
    # Get sample of high PageRank nodes and their neighbors
    top_pr_nodes = top_pagerank.limit(50).select("id").toPandas()['id'].tolist()
    
    # Get edges involving these nodes
    pr_edges = g.edges.filter(
        col("src").isin(top_pr_nodes) | col("dst").isin(top_pr_nodes)
    ).limit(200).toPandas()
    
    # Get all nodes involved
    pr_node_ids = list(set(pr_edges['src'].tolist() + pr_edges['dst'].tolist()))
    pr_vertices = pagerank_results.vertices.filter(col("id").isin(pr_node_ids)).toPandas()
    
    # Create NetworkX graph
    G_pr = nx.DiGraph()
    
    for _, row in pr_vertices.iterrows():
        G_pr.add_node(row['id'],
                     pagerank=row['pagerank'],
                     node_type=row['type'],
                     fraud_score=row['fraud_score'] if pd.notna(row['fraud_score']) else 0)
    
    for _, row in pr_edges.iterrows():
        if row['src'] in G_pr.nodes() and row['dst'] in G_pr.nodes():
            G_pr.add_edge(row['src'], row['dst'])
    
    # Visualization
    fig, ax = plt.subplots(figsize=(20, 16))
    pos = nx.spring_layout(G_pr, k=3, iterations=50, seed=42)
    
    # Node sizes by PageRank (scaled up for visibility)
    node_sizes = [G_pr.nodes[n]['pagerank'] * 50000 + 200 for n in G_pr.nodes()]
    
    # Node colors by fraud score
    node_colors = [G_pr.nodes[n].get('fraud_score', 0.5) for n in G_pr.nodes()]
    
    nx.draw_networkx_nodes(G_pr, pos, node_size=node_sizes, node_color=node_colors,
                          cmap='RdYlGn_r', alpha=0.7, vmin=0, vmax=1)
    nx.draw_networkx_edges(G_pr, pos, edge_color='gray', arrows=True,
                          arrowsize=10, alpha=0.3, width=1.5)
    
    labels = {node: node[:8] if len(node) > 8 else node for node in G_pr.nodes()}
    nx.draw_networkx_labels(G_pr, pos, labels, font_size=7)
    
    # Color bar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Fraud Score', rotation=270, labelpad=20)
    
    plt.title(f"PageRank Visualization: Node Size = Influence (PageRank)\n" +
             "Color = Fraud Score | Larger nodes = More influential entities",
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Showing {len(G_pr.nodes())} nodes - Larger nodes have higher PageRank scores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Community Detection with Label Propagation
# MAGIC 
# MAGIC Discover communities within the network - may reveal sub-groups of fraud rings

# COMMAND ----------

print("Running Label Propagation for community detection...")
lp_results = g.labelPropagation(maxIter=5)

# Analyze communities
communities = lp_results.select("id", "type", "fraud_score", "label")

community_stats = communities.groupBy("label").agg(
    count("*").alias("size"),
    sum(when(col("type") == "claim", 1).otherwise(0)).alias("claim_count"),
    sum(when(col("type") == "policyholder", 1).otherwise(0)).alias("policyholder_count"),
    round(avg(when(col("type") == "claim", col("fraud_score"))), 3).alias("avg_fraud_score"),
    sum(when((col("type") == "claim") & (col("fraud_score") >= fraud_threshold), 1).otherwise(0)).alias("high_fraud_claims")
).withColumn(
    "fraud_concentration",
    when(col("claim_count") > 0, 
         round(col("high_fraud_claims") / col("claim_count") * 100, 1)
    ).otherwise(0.0)
).filter(col("claim_count") >= 2).orderBy(desc("avg_fraud_score"))

print(f"\n🏘️ Discovered {community_stats.count():,} communities with claims")
print("\nTop 20 communities by average fraud score:")
display(community_stats.limit(20))

# High-risk communities
high_risk_communities = community_stats.filter(
    (col("avg_fraud_score") >= 0.6) | (col("fraud_concentration") >= 40)
).orderBy(desc("fraud_concentration"))

print(f"\n🚨 High-Risk Communities: {high_risk_communities.count():,}")
display(high_risk_communities)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Visualization: Community Detection (Label Propagation)
# MAGIC 
# MAGIC Visualize communities with different colors

# COMMAND ----------

# Visualize communities
print("Creating community detection visualization...")

# Get top high-risk communities
top_communities = high_risk_communities.limit(5).select("label").toPandas()['label'].tolist()

# Get vertices in these communities
comm_vertices = communities.filter(col("label").isin(top_communities)).limit(150).toPandas()
comm_vertex_ids = comm_vertices['id'].tolist()

comm_edges = g.edges.filter(
    col("src").isin(comm_vertex_ids) & col("dst").isin(comm_vertex_ids)
).limit(200).toPandas()

# Create NetworkX graph
G_comm = nx.DiGraph()

for _, row in comm_vertices.iterrows():
    G_comm.add_node(row['id'],
                   community=row['label'],
                   node_type=row['type'],
                   fraud_score=row['fraud_score'] if pd.notna(row['fraud_score']) else 0)

for _, row in comm_edges.iterrows():
    if row['src'] in G_comm.nodes() and row['dst'] in G_comm.nodes():
        G_comm.add_edge(row['src'], row['dst'])

# Visualization
fig, ax = plt.subplots(figsize=(20, 16))
pos = nx.spring_layout(G_comm, k=3, iterations=50, seed=42)

# Color by community
comm_colors = [G_comm.nodes[n]['community'] for n in G_comm.nodes()]
unique_comms = list(set(comm_colors))
color_map_comm = {comm: i for i, comm in enumerate(unique_comms)}
node_colors_comm = [color_map_comm[G_comm.nodes[n]['community']] for n in G_comm.nodes()]

# Node sizes by fraud score
node_sizes_comm = [G_comm.nodes[n].get('fraud_score', 0.5) * 1000 + 300 for n in G_comm.nodes()]

nx.draw_networkx_nodes(G_comm, pos, node_color=node_colors_comm, node_size=node_sizes_comm,
                      cmap='Set3', alpha=0.8)
nx.draw_networkx_edges(G_comm, pos, edge_color='gray', arrows=True,
                      arrowsize=10, alpha=0.3, width=1.5)

labels_comm = {node: node[:8] if len(node) > 8 else node for node in G_comm.nodes()}
nx.draw_networkx_labels(G_comm, pos, labels_comm, font_size=6)

plt.title(f"Community Detection Visualization (Top {len(unique_comms)} High-Risk Communities)\n" +
         "Each color = Different community | Node size = Fraud score",
         fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"\n📊 Showing {len(G_comm.nodes())} nodes across {len(unique_comms)} communities")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Triangle Count - Measure Network Density
# MAGIC 
# MAGIC High triangle counts indicate tightly-connected groups (potential collusion)

# COMMAND ----------

print("Running Triangle Count...")
triangle_results = g.triangleCount()

# Analyze triangle counts
# Note: triangleCount() returns a DataFrame directly, not a GraphFrame
triangle_analysis = triangle_results.select(
    "id", "type", "fraud_score", "count"
).orderBy(desc("count"))

print("\n🔺 Top 20 Entities by Triangle Count:")
display(triangle_analysis.limit(20))

# Focus on high-fraud entities with many triangles
high_fraud_triangles = triangle_analysis.filter(
    (col("type") == "claim") & (col("fraud_score") >= fraud_threshold) & (col("count") > 0)
).orderBy(desc("count"))

print(f"\n🚨 High-fraud claims with triangles (collusion indicators): {high_fraud_triangles.count():,}")
display(high_fraud_triangles.limit(20))

# Average triangles by entity type
avg_triangles = triangle_results.groupBy("type").agg(
    avg("count").alias("avg_triangle_count"),
    max("count").alias("max_triangle_count")
).orderBy(desc("avg_triangle_count"))

print("\n📊 Average Triangle Count by Entity Type:")
display(avg_triangles)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Visualization: Triangle Count
# MAGIC 
# MAGIC Visualize entities with high triangle counts (dense local networks)

# COMMAND ----------

# Visualize triangle count results
print("Creating triangle count visualization...")

# Get entities with high triangle counts
high_triangle_nodes = triangle_analysis.filter(col("count") > 0).limit(100).toPandas()
ht_node_ids = high_triangle_nodes['id'].tolist()

ht_edges = g.edges.filter(
    col("src").isin(ht_node_ids) & col("dst").isin(ht_node_ids)
).limit(200).toPandas()

# Create NetworkX graph
G_tri = nx.Graph()  # Undirected for triangle visualization

for _, row in high_triangle_nodes.iterrows():
    G_tri.add_node(row['id'],
                  triangles=row['count'],
                  node_type=row['type'],
                  fraud_score=row['fraud_score'] if pd.notna(row['fraud_score']) else 0)

for _, row in ht_edges.iterrows():
    if row['src'] in G_tri.nodes() and row['dst'] in G_tri.nodes():
        G_tri.add_edge(row['src'], row['dst'])

# Visualization
fig, ax = plt.subplots(figsize=(20, 16))
pos = nx.spring_layout(G_tri, k=2, iterations=50, seed=42)

# Node sizes by triangle count
node_sizes_tri = [G_tri.nodes[n]['triangles'] * 100 + 300 for n in G_tri.nodes()]

# Node colors by fraud score
node_colors_tri = [G_tri.nodes[n].get('fraud_score', 0.5) for n in G_tri.nodes()]

nx.draw_networkx_nodes(G_tri, pos, node_size=node_sizes_tri, node_color=node_colors_tri,
                      cmap='RdYlGn_r', alpha=0.7, vmin=0, vmax=1)
nx.draw_networkx_edges(G_tri, pos, edge_color='gray', alpha=0.4, width=2)

labels_tri = {node: node[:8] if len(node) > 8 else node for node in G_tri.nodes()}
nx.draw_networkx_labels(G_tri, pos, labels_tri, font_size=6)

# Color bar
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Fraud Score', rotation=270, labelpad=20)

plt.title(f"Triangle Count Visualization: Node Size = # of Triangles\n" +
         "Larger nodes participate in more triangles (higher clustering/collusion)",
         fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"\n📊 Showing {len(G_tri.nodes())} nodes with triangles - Larger = More triangles (collusion indicator)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Motif Finding - Discover Suspicious Patterns

# COMMAND ----------

print("Running Motif Finding to discover suspicious patterns...")

# Pattern 1: Multiple claims from same policyholder with high fraud scores
print("\n🔍 Pattern 1: Policyholders with multiple high-fraud claims")
motif1 = g.find("(c1)-[e1]->(p); (c2)-[e2]->(p)").filter(
    "(c1.type = 'claim') AND (c2.type = 'claim') AND (p.type = 'policyholder') AND " +
    "(c1.id < c2.id) AND (c1.fraud_score >= 0.7) AND (c2.fraud_score >= 0.7)"
)

if motif1.count() > 0:
    pattern1_summary = motif1.groupBy("p.id").agg(
        count("*").alias("high_fraud_claim_pairs"),
        avg("c1.fraud_score").alias("avg_fraud_score")
    ).orderBy(desc("high_fraud_claim_pairs"))
    
    print(f"Found {pattern1_summary.count():,} policyholders with multiple high-fraud claims")
    display(pattern1_summary.limit(20))
else:
    print("No pattern found")

# Pattern 2: Shared policyholders filing high-fraud claims
print("\n🔍 Pattern 2: Connected policyholders (shared contacts) both filing high-fraud claims")
# Note: Use -> for directed edge (GraphFrames requires explicit direction)
motif2 = g.find("(c1)-[e1]->(p1); (c2)-[e2]->(p2); (p1)-[e3]->(p2)").filter(
    "(c1.type = 'claim') AND (c2.type = 'claim') AND " +
    "(p1.type = 'policyholder') AND (p2.type = 'policyholder') AND " +
    "(c1.id < c2.id) AND (p1.id < p2.id) AND " +
    "(c1.fraud_score >= 0.7) AND (c2.fraud_score >= 0.7) AND " +
    "(e3.relationship = 'shared_contact')"
)

print(f"Found {motif2.count():,} instances of connected policyholders with high-fraud claims")
if motif2.count() > 0:
    print("\nSample suspicious patterns:")
    display(motif2.select(
        col("c1.id").alias("claim1"),
        col("c1.fraud_score").alias("score1"),
        col("p1.id").alias("policyholder1"),
        col("c2.id").alias("claim2"),
        col("c2.fraud_score").alias("score2"),
        col("p2.id").alias("policyholder2")
    ).limit(20))

# Pattern 3: Same adjuster handling multiple high-fraud claims
print("\n🔍 Pattern 3: Adjusters handling multiple high-fraud claims")
adjuster_fraud_pattern = claims.filter(col("fraud_score") >= fraud_threshold).groupBy("adjuster_id").agg(
    count("*").alias("high_fraud_claim_count"),
    round(avg("fraud_score"), 3).alias("avg_fraud_score"),
    round(sum("claim_amount"), 2).alias("total_amount")
).filter(col("high_fraud_claim_count") >= 3).orderBy(desc("high_fraud_claim_count"))

print(f"Found {adjuster_fraud_pattern.count():,} adjusters with 3+ high-fraud claims")
if adjuster_fraud_pattern.count() > 0:
    print("\n⚠️ Adjusters to investigate:")
    display(adjuster_fraud_pattern.limit(20))

# Pattern 4: Same service provider in multiple high-fraud claims
print("\n🔍 Pattern 4: Service providers in multiple high-fraud claims")
provider_fraud_pattern = claims.filter(col("fraud_score") >= fraud_threshold).groupBy("service_provider_id").agg(
    count("*").alias("high_fraud_claim_count"),
    round(avg("fraud_score"), 3).alias("avg_fraud_score"),
    round(sum("claim_amount"), 2).alias("total_amount")
).filter(col("high_fraud_claim_count") >= 3).orderBy(desc("high_fraud_claim_count"))

print(f"Found {provider_fraud_pattern.count():,} providers with 3+ high-fraud claims")
if provider_fraud_pattern.count() > 0:
    print("\n⚠️ Service providers to investigate:")
    display(provider_fraud_pattern.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Shortest Paths from High-Fraud Claims

# COMMAND ----------

# Get top high-fraud claims as landmarks
high_fraud_claims = claims.filter(col("fraud_score") >= fraud_threshold).orderBy(desc("fraud_score")).limit(5)

if high_fraud_claims.count() > 0:
    landmarks = [row.claim_id for row in high_fraud_claims.collect()]
    
    print(f"Finding shortest paths from {len(landmarks)} high-fraud claims:")
    for claim_id in landmarks:
        fraud_score = claims.filter(col("claim_id") == claim_id).select("fraud_score").first()[0]
        print(f"  • {claim_id} (score: {fraud_score:.3f})")
    
    sp_results = g.shortestPaths(landmarks=landmarks)
    
    # Find entities close to multiple high-fraud claims
    close_to_fraud = sp_results.select(
        "id", "type", "fraud_score", "distances"
    ).filter(size(col("distances")) > 0).withColumn(
        "num_connections",
        size(col("distances"))
    ).withColumn(
        "min_distance",
        expr("aggregate(map_values(distances), 999, (acc, x) -> CASE WHEN x < acc THEN x ELSE acc END)")
    ).filter(
        (col("num_connections") >= 2) & (col("min_distance") <= 3)
    ).orderBy(desc("num_connections"), "min_distance")
    
    print(f"\n⚠️ Entities close to multiple high-fraud claims: {close_to_fraud.count():,}")
    display(close_to_fraud.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC # 📊 Summary: Discovered Fraud Rings

# COMMAND ----------

print("="*80)
print("FRAUD RING DISCOVERY SUMMARY")
print("="*80)

# Overall statistics
total_claims = claims.count()
high_fraud_claims_count = claims.filter(col("fraud_score") >= fraud_threshold).count()

print(f"\n📊 DATASET:")
print(f"   Total claims: {total_claims:,}")
print(f"   High fraud score claims (≥{fraud_threshold}): {high_fraud_claims_count:,} ({high_fraud_claims_count/total_claims*100:.1f}%)")

# Discovered fraud rings
fraud_rings_count = suspected_fraud_rings.count() if suspected_fraud_rings.count() > 0 else 0
print(f"\n🚨 DISCOVERED FRAUD RINGS: {fraud_rings_count}")

if fraud_rings_count > 0:
    # Summary stats
    fraud_ring_stats = suspected_fraud_rings.agg(
        sum("claim_count").alias("total_claims_in_rings"),
        sum("policyholder_count").alias("total_policyholders_in_rings"),
        round(avg("avg_fraud_score"), 3).alias("avg_fraud_score_across_rings"),
        round(avg("fraud_concentration"), 1).alias("avg_fraud_concentration")
    ).collect()[0]
    
    print(f"   Total claims in fraud rings: {fraud_ring_stats.total_claims_in_rings:,}")
    print(f"   Total policyholders in fraud rings: {fraud_ring_stats.total_policyholders_in_rings:,}")
    print(f"   Average fraud score: {fraud_ring_stats.avg_fraud_score_across_rings}")
    print(f"   Average fraud concentration: {fraud_ring_stats.avg_fraud_concentration}%")
    
    print(f"\n🏆 TOP 3 FRAUD RINGS:")
    top_3_rings = suspected_fraud_rings.limit(3).collect()
    for i, ring in enumerate(top_3_rings, 1):
        print(f"\n   Ring #{i}:")
        print(f"     Claims: {ring.claim_count}")
        print(f"     Policyholders: {ring.policyholder_count}")
        print(f"     Avg fraud score: {ring.avg_fraud_score}")
        print(f"     Fraud concentration: {ring.fraud_concentration}%")

# Suspicious patterns found
print(f"\n🔍 SUSPICIOUS PATTERNS:")
if 'pattern1_summary' in locals() and pattern1_summary.count() > 0:
    print(f"   Policyholders with multiple high-fraud claims: {pattern1_summary.count():,}")
if motif2.count() > 0:
    print(f"   Connected policyholders filing high-fraud claims: {motif2.count():,}")
if adjuster_fraud_pattern.count() > 0:
    print(f"   Adjusters handling 3+ high-fraud claims: {adjuster_fraud_pattern.count():,}")
if provider_fraud_pattern.count() > 0:
    print(f"   Providers in 3+ high-fraud claims: {provider_fraud_pattern.count():,}")

print(f"\n💡 KEY INSIGHTS:")
print(f"   ✅ Fraud rings discovered through graph analysis (NOT pre-labeled)")
print(f"   ✅ Connected components identified suspicious networks")
print(f"   ✅ PageRank found influential entities (potential ringleaders)")
print(f"   ✅ Motif finding revealed coordinated fraud patterns")
print(f"   ✅ Community detection segmented fraud groups")

print(f"\n📈 NEXT STEPS:")
print(f"   1. Investigate top fraud rings identified in 'discovered_fraud_rings' table")
print(f"   2. Review adjusters/providers handling multiple high-fraud claims")
print(f"   3. Analyze shared contact patterns (addresses, phones)")
print(f"   4. Use BFS to expand investigation from confirmed fraud cases")
print(f"   5. Set up monitoring for new claims in discovered fraud rings")

print("\n" + "="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC # 🎯 Key Takeaways
# MAGIC 
# MAGIC ## What We Accomplished:
# MAGIC 
# MAGIC 1. **Started with neutral data**: Only fraud_scores, no pre-labeled fraud rings
# MAGIC 2. **Built a graph**: Claims, policyholders, adjusters, service providers
# MAGIC 3. **Used GraphFrames to DISCOVER**:
# MAGIC    - **Fraud rings** through connected components + fraud score analysis
# MAGIC    - **Ringleaders** through PageRank on suspicious networks
# MAGIC    - **Fraud communities** through label propagation
# MAGIC    - **Collusion patterns** through triangle counting
# MAGIC    - **Suspicious relationships** through motif finding
# MAGIC 
# MAGIC ## Why GraphFrames is Powerful:
# MAGIC 
# MAGIC - ✅ **Discovers hidden patterns** that rule-based systems miss
# MAGIC - ✅ **Scales to millions** of claims and relationships
# MAGIC - ✅ **Combines network structure with fraud scores** for accurate detection
# MAGIC - ✅ **Multiple algorithms** provide different perspectives on fraud
# MAGIC - ✅ **Production-ready** for real-time fraud detection
# MAGIC 
# MAGIC ## The Magic:
# MAGIC 
# MAGIC **We didn't tell the system which networks were fraud rings.**  
# MAGIC **GraphFrames discovered them by analyzing:**
# MAGIC - Network topology (who's connected to whom)
# MAGIC - Fraud score patterns (which claims are suspicious)
# MAGIC - Shared attributes (addresses, phones, service providers)
# MAGIC - Temporal patterns (when claims were filed)
# MAGIC 
# MAGIC This is the power of **graph-based fraud detection**! 🚀
