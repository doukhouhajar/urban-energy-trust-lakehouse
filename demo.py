import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum, desc, expr,
    date_trunc, when, min as spark_min, max as spark_max
)
from src.utils.config import load_config
from src.utils.spark_session import get_or_create_spark_session
import os

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(spark, config):   
    paths = config['paths']
    
    print("Loading data from Gold layer...")
    
    data = {
        'quality_scores': spark.read.format("delta").load(
            f"{paths['gold_root']}/quality_scores"
        ),
        'quality_incidents': spark.read.format("delta").load(
            f"{paths['gold_root']}/quality_incidents"
        ),
        'household_info': spark.read.format("delta").load(
            f"{paths['bronze_root']}/household_info"
        ),
        'consumption_analytics': spark.read.format("delta").load(
            f"{paths['gold_root']}/consumption_analytics"
        )
    }
    
    # Try to load predictions if they exist
    try:
        data['predictions'] = spark.read.format("delta").load(
            f"{paths['gold_root']}/quality_risk_predictions"
        )
    except:
        print("Note: Quality risk predictions not found. Run ML model training first.")
        data['predictions'] = None
    
    return data


def demo_1_quality_score_distribution(data, output_dir="demonstrations/plots"):
    print("DEMONSTRATION 1: Quality Score Distribution")

    
    scores_df = data['quality_scores']
    
    # Convert to Pandas for plotting
    scores_pd = scores_df.select("quality_score", "quality_category").toPandas()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality Score Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram of quality scores
    axes[0, 0].hist(scores_pd['quality_score'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(70, color='red', linestyle='--', label='Low Quality Threshold (70)')
    axes[0, 0].set_xlabel('Quality Score (0-100)')
    axes[0, 0].set_ylabel('Number of Household-Days')
    axes[0, 0].set_title('Distribution of Quality Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Quality category pie chart
    category_counts = scores_pd['quality_category'].value_counts()
    axes[0, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Quality Score Categories')
    
    # 3. Box plot by category
    scores_pd.boxplot(column='quality_score', by='quality_category', ax=axes[1, 0])
    axes[1, 0].set_title('Quality Score Distribution by Category')
    axes[1, 0].set_xlabel('Quality Category')
    axes[1, 0].set_ylabel('Quality Score')
    axes[1, 0].get_figure().suptitle('')  # Remove default title
    
    # 4. Statistics summary
    stats = scores_pd['quality_score'].describe()
    axes[1, 1].axis('off')
    stats_text = f"""
    Quality Score Statistics:
    
    Mean: {stats['mean']:.2f}
    Median: {stats['50%']:.2f}
    Std Dev: {stats['std']:.2f}
    Min: {stats['min']:.2f}
    Max: {stats['max']:.2f}
    
    Total Records: {len(scores_pd):,}
    Low Quality (<70): {(scores_pd['quality_score'] < 70).sum():,}
    Low Quality %: {(scores_pd['quality_score'] < 70).sum() / len(scores_pd) * 100:.1f}%
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/01_quality_score_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/01_quality_score_distribution.png")
    
    # Show top 10 worst quality households
    print("\nTop 10 Households with Lowest Quality Scores:")
    worst_quality = scores_df.groupBy("household_id").agg(
        avg("quality_score").alias("avg_quality_score"),
        count("*").alias("days_with_data")
    ).orderBy("avg_quality_score").limit(10)
    worst_quality.show(truncate=False)
    
    return fig


def demo_2_low_quality_households(data, output_dir="demonstrations/plots"):
    print("DEMONSTRATION 2: Households with Low Quality Data")

    
    scores_df = data['quality_scores']
    household_info = data['household_info']
    
    # Join with household info
    low_quality = scores_df.filter(col("quality_score") < 70) \
        .join(household_info.select("household_id", "acorn_group", "tariff_type"), 
              on="household_id", how="left") \
        .groupBy("household_id", "acorn_group", "tariff_type") \
        .agg(
            avg("quality_score").alias("avg_quality_score"),
            spark_min("quality_score").alias("min_quality_score"),
            count("*").alias("low_quality_days")
        ) \
        .orderBy(desc("low_quality_days"), "avg_quality_score") \
        .limit(20)
    
    print("\nTop 20 Households with Most Low Quality Days:")
    low_quality.show(truncate=False)
    
    # Convert to Pandas for visualization
    low_quality_pd = low_quality.toPandas()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Low Quality Households Analysis', fontsize=16, fontweight='bold')
    
    # 1. Low quality days by ACORN group
    if 'acorn_group' in low_quality_pd.columns:
        acorn_summary = low_quality_pd.groupby('acorn_group').agg({
            'low_quality_days': 'sum',
            'household_id': 'count'
        }).reset_index()
        acorn_summary.columns = ['acorn_group', 'total_low_quality_days', 'household_count']
        
        axes[0].barh(acorn_summary['acorn_group'], acorn_summary['total_low_quality_days'])
        axes[0].set_xlabel('Total Low Quality Days')
        axes[0].set_title('Low Quality Days by ACORN Group')
        axes[0].grid(True, alpha=0.3, axis='x')
    
    # 2. Quality score vs low quality days scatter
    axes[1].scatter(low_quality_pd['low_quality_days'], low_quality_pd['avg_quality_score'], 
                   alpha=0.6, s=100)
    axes[1].axhline(70, color='red', linestyle='--', label='Quality Threshold')
    axes[1].set_xlabel('Number of Low Quality Days')
    axes[1].set_ylabel('Average Quality Score')
    axes[1].set_title('Quality Score vs Low Quality Days')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_low_quality_households.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/02_low_quality_households.png")
    
    return fig, low_quality


def demo_3_incidents_analysis(data, output_dir="demonstrations/plots"):
    print("DEMONSTRATION 3: Quality Incidents by Type and Severity")

    
    incidents_df = data['quality_incidents']
    
    # Incidents by type
    incidents_by_type = incidents_df.groupBy("rule_name", "severity") \
        .agg(count("*").alias("incident_count"),
             count("entity_id").alias("affected_entities")) \
        .orderBy(desc("incident_count"))
    
    print("\nIncidents by Type and Severity:")
    incidents_by_type.show(truncate=False)
    
    # Convert to Pandas
    incidents_pd = incidents_by_type.toPandas()
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Quality Incidents Analysis', fontsize=16, fontweight='bold')
    
    # 1. Incidents by type (bar chart)
    if len(incidents_pd) > 0:
        incidents_pivot = incidents_pd.pivot(index='rule_name', columns='severity', values='incident_count').fillna(0)
        incidents_pivot.plot(kind='barh', ax=axes[0], stacked=True)
        axes[0].set_xlabel('Number of Incidents')
        axes[0].set_title('Incidents by Type and Severity')
        axes[0].legend(title='Severity')
        axes[0].grid(True, alpha=0.3, axis='x')
    
    # 2. Timeline of incidents (if timestamp available)
    try:
        incidents_timeline = incidents_df.groupBy(
            date_trunc("day", col("incident_timestamp")).alias("day")
        ).agg(count("*").alias("daily_incidents")).orderBy("day")
        
        timeline_pd = incidents_timeline.toPandas()
        if len(timeline_pd) > 0:
            timeline_pd['day'] = pd.to_datetime(timeline_pd['day'])
            axes[1].plot(timeline_pd['day'], timeline_pd['daily_incidents'], marker='o', linewidth=2)
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Daily Incidents')
            axes[1].set_title('Quality Incidents Over Time')
            axes[1].grid(True, alpha=0.3)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Timeline data not available\n{str(e)}', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Quality Incidents Timeline')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_incidents_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/03_incidents_analysis.png")
    
    return fig


def demo_4_quality_risk_predictions(data, output_dir="demonstrations/plots"):
    print("DEMONSTRATION 4: Quality Risk Predictions (ML Model)")

    
    if data['predictions'] is None:
        print("No predictions available. Please train the ML model first:")
        print("  python -m src.ml.train_quality_model")
        print("  python -m src.ml.predict_quality_risk")
        return None
    
    predictions_df = data['predictions']
    household_info = data['household_info']
    
    # Join with household info
    high_risk = predictions_df.filter(col("risk_score") >= 0.7) \
        .join(household_info.select("household_id", "acorn_group"), 
              on="household_id", how="left") \
        .orderBy(desc("risk_score")) \
        .limit(50)
    
    print("\nTop 50 Households at High Risk (Risk Score >= 0.7):")
    high_risk.show(truncate=False)
    
    # Convert to Pandas
    predictions_pd = predictions_df.toPandas()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quality Risk Predictions Analysis', fontsize=16, fontweight='bold')
    
    # 1. Risk score distribution
    axes[0, 0].hist(predictions_pd['risk_score'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0.7, color='red', linestyle='--', label='High Risk Threshold (0.7)')
    axes[0, 0].axvline(0.3, color='orange', linestyle='--', label='Medium Risk Threshold (0.3)')
    axes[0, 0].set_xlabel('Risk Score (0-1)')
    axes[0, 0].set_ylabel('Number of Households')
    axes[0, 0].set_title('Risk Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Risk categories pie chart
    if 'risk_category' in predictions_pd.columns:
        risk_counts = predictions_pd['risk_category'].value_counts()
        axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Households by Risk Category')
    
    # 3. Top 20 high risk households
    top_risk = predictions_pd.nlargest(20, 'risk_score')
    axes[1, 0].barh(range(len(top_risk)), top_risk['risk_score'])
    axes[1, 0].set_yticks(range(len(top_risk)))
    axes[1, 0].set_yticklabels(top_risk['household_id'].values, fontsize=8)
    axes[1, 0].set_xlabel('Risk Score')
    axes[1, 0].set_title('Top 20 High Risk Households')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Statistics
    axes[1, 1].axis('off')
    stats = predictions_pd['risk_score'].describe()
    risk_stats_text = f"""
    Risk Prediction Statistics:
    
    Mean Risk: {stats['mean']:.3f}
    Median Risk: {stats['50%']:.3f}
    Max Risk: {stats['max']:.3f}
    
    Total Predictions: {len(predictions_pd):,}
    High Risk (>=0.7): {(predictions_pd['risk_score'] >= 0.7).sum():,}
    Medium Risk (0.3-0.7): {((predictions_pd['risk_score'] >= 0.3) & (predictions_pd['risk_score'] < 0.7)).sum():,}
    Low Risk (<0.3): {(predictions_pd['risk_score'] < 0.3).sum():,}
    """
    axes[1, 1].text(0.1, 0.5, risk_stats_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_quality_risk_predictions.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/04_quality_risk_predictions.png")
    
    return fig, high_risk


def demo_5_quality_trends_over_time(data, output_dir="demonstrations/plots"):
    print("DEMONSTRATION 5: Quality Score Trends Over Time")

    
    scores_df = data['quality_scores']
    
    # Aggregate by day
    daily_quality = scores_df.groupBy(
        date_trunc("day", col("score_date")).alias("day")
    ).agg(
        avg("quality_score").alias("avg_quality_score"),
        spark_min("quality_score").alias("min_quality_score"),
        spark_max("quality_score").alias("max_quality_score"),
        count("*").alias("household_days")
    ).orderBy("day")
    
    print("\nDaily Quality Score Summary:")
    daily_quality.show(truncate=False)
    
    # Convert to Pandas
    daily_pd = daily_quality.toPandas()
    daily_pd['day'] = pd.to_datetime(daily_pd['day'])
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Quality Score Trends Over Time', fontsize=16, fontweight='bold')
    
    # 1. Average quality over time with min/max bands
    axes[0].plot(daily_pd['day'], daily_pd['avg_quality_score'], 
                label='Average Quality Score', linewidth=2, marker='o')
    axes[0].fill_between(daily_pd['day'], daily_pd['min_quality_score'], 
                        daily_pd['max_quality_score'], alpha=0.3, label='Min-Max Range')
    axes[0].axhline(70, color='red', linestyle='--', label='Low Quality Threshold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Quality Score')
    axes[0].set_title('Daily Average Quality Score with Range')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Number of household-days over time
    axes[1].bar(daily_pd['day'], daily_pd['household_days'], alpha=0.7, width=0.8)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Number of Household-Days')
    axes[1].set_title('Data Coverage Over Time')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_quality_trends_over_time.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/05_quality_trends_over_time.png")
    
    return fig


def demo_6_acorn_group_analysis(data, output_dir="demonstrations/plots"):
    """Demonstration 6: Quality by ACORN Socio-Economic Groups"""
    print("DEMONSTRATION 6: Quality Scores by ACORN Group")

    
    scores_df = data['quality_scores']
    household_info = data['household_info']
    
    # Join and aggregate by ACORN group
    acorn_quality = scores_df.join(
        household_info.select("household_id", "acorn_group"),
        on="household_id", how="left"
    ).groupBy("acorn_group").agg(
        avg("quality_score").alias("avg_quality_score"),
        spark_min("quality_score").alias("min_quality_score"),
        spark_max("quality_score").alias("max_quality_score"),
        count("*").alias("total_records"),
        spark_sum(when(col("quality_score") < 70, 1).otherwise(0)).alias("low_quality_count")
    ).withColumn(
        "low_quality_rate",
        col("low_quality_count") / col("total_records")
    ).orderBy("avg_quality_score")
    
    print("\nQuality Scores by ACORN Group:")
    acorn_quality.show(truncate=False)
    
    # Convert to Pandas
    acorn_pd = acorn_quality.toPandas()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Quality Analysis by ACORN Socio-Economic Groups', fontsize=16, fontweight='bold')
    
    # 1. Average quality by ACORN group
    axes[0].barh(acorn_pd['acorn_group'], acorn_pd['avg_quality_score'])
    axes[0].axvline(70, color='red', linestyle='--', label='Low Quality Threshold')
    axes[0].set_xlabel('Average Quality Score')
    axes[0].set_title('Average Quality Score by ACORN Group')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 2. Low quality rate by ACORN group
    axes[1].barh(acorn_pd['acorn_group'], acorn_pd['low_quality_rate'] * 100)
    axes[1].set_xlabel('Low Quality Rate (%)')
    axes[1].set_title('Low Quality Rate by ACORN Group')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_acorn_group_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/06_acorn_group_analysis.png")
    
    return fig


def demo_7_component_breakdown(data, output_dir="demonstrations/plots"):
    """Demonstration 7: Quality Score Component Breakdown"""
    print("DEMONSTRATION 7: Quality Score Component Analysis")

    
    scores_df = data['quality_scores']
    
    # Calculate average of each component
    components = scores_df.agg(
        avg("completeness_score").alias("avg_completeness"),
        avg("temporal_score").alias("avg_temporal"),
        avg("business_score").alias("avg_business"),
        avg("schema_score").alias("avg_schema")
    ).collect()[0]
    
    print("\nAverage Component Scores (0-1 scale):")
    print(f"  Completeness: {components['avg_completeness']:.3f}")
    print(f"  Temporal Coherence: {components['avg_temporal']:.3f}")
    print(f"  Business Rules: {components['avg_business']:.3f}")
    print(f"  Schema Validity: {components['avg_schema']:.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Quality Score Component Breakdown', fontsize=16, fontweight='bold')
    
    # 1. Component scores bar chart
    component_names = ['Completeness', 'Temporal', 'Business Rules', 'Schema']
    component_values = [
        components['avg_completeness'],
        components['avg_temporal'],
        components['avg_business'],
        components['avg_schema']
    ]
    component_weights = [0.40, 0.25, 0.20, 0.15]  # From config
    
    bars = axes[0].bar(component_names, component_values, alpha=0.7)
    axes[0].axhline(1.0, color='green', linestyle='--', label='Perfect Score')
    axes[0].set_ylabel('Average Score (0-1)')
    axes[0].set_title('Average Component Scores')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add weight labels
    for i, (bar, weight) in enumerate(zip(bars, component_weights)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'Weight: {weight*100}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Contribution to overall score
    contributions = [v * w * 100 for v, w in zip(component_values, component_weights)]
    axes[1].bar(component_names, contributions, alpha=0.7, color='coral')
    axes[1].set_ylabel('Contribution to Overall Score')
    axes[1].set_title('Component Contribution to Overall Quality Score')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (name, contrib) in enumerate(zip(component_names, contributions)):
        axes[1].text(i, contrib + 0.5, f'{contrib:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_component_breakdown.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/07_component_breakdown.png")
    
    return fig


def demo_8_pipeline_summary(spark, data, config, output_dir="demonstrations/plots"):
    """Demonstration 8: Pipeline Execution Summary"""
    print("DEMONSTRATION 8: Pipeline Execution Summary")

    
    # Load audit log
    try:
        audit_log = spark.read.format("delta").load(
            f"{config['paths']['gold_root']}/audit_log"
        )
        
        latest_run = audit_log.orderBy(desc("run_timestamp")).limit(1).collect()[0]
        
        print("\nLatest Pipeline Run Summary:")
        print(f"  Pipeline: {latest_run['pipeline_name']}")
        print(f"  Status: {latest_run['status']}")
        print(f"  Timestamp: {latest_run['run_timestamp']}")
        
        # Parse row counts from JSON
        import json
        row_counts = json.loads(latest_run['row_counts'])
        quality_summary = json.loads(latest_run['quality_score_summary'])
        
        print("\nTable Row Counts:")
        for table, count in row_counts.items():
            print(f"  {table}: {count:,}")
        
        print("\nQuality Summary:")
        for key, value in quality_summary.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Could not load audit log: {e}")


def run_all_demonstrations():
    
    config = load_config()
    spark = get_or_create_spark_session(config, use_docker=False)
    
    try:
        # Load all data
        data = load_data(spark, config)
        
        # Create output directory
        output_dir = "demonstrations/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run all demonstrations
        demo_1_quality_score_distribution(data, output_dir)
        demo_2_low_quality_households(data, output_dir)
        demo_3_incidents_analysis(data, output_dir)
        demo_4_quality_risk_predictions(data, output_dir)
        demo_5_quality_trends_over_time(data, output_dir)
        demo_6_acorn_group_analysis(data, output_dir)
        demo_7_component_breakdown(data, output_dir)
        demo_8_pipeline_summary(spark, data, config, output_dir)
        

        print("ALL DEMONSTRATIONS COMPLETED!")
    
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()


if __name__ == "__main__":
    run_all_demonstrations()