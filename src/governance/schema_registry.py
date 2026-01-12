from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField
from delta.tables import DeltaTable
from typing import Dict, Optional
import os


def handle_schema_evolution(
    spark: SparkSession,
    new_df: DataFrame,
    target_path: str, # Target Delta table path
    mode: str = "merge", # Evolution mode - "merge" (add new columns), "fail" (raise error), "overwrite" (replace schema)
    allow_additional_columns: bool = True # If True, new columns in new_df are allowed
) -> bool:
    if not os.path.exists(target_path) or not DeltaTable.isDeltaTable(spark, target_path):
        new_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
        return True
    
    existing_df = spark.read.format("delta").load(target_path)
    existing_schema = existing_df.schema
    new_schema = new_df.schema
    
    existing_fields = {f.name: f for f in existing_schema.fields}
    new_fields = {f.name: f for f in new_schema.fields}
    
    removed_columns = set(existing_fields.keys()) - set(new_fields.keys())
    if removed_columns and mode != "overwrite":
        if mode == "fail":
            raise ValueError(f"Schema evolution failed: columns removed: {removed_columns}")
    
    new_columns = set(new_fields.keys()) - set(existing_fields.keys())
    if new_columns and not allow_additional_columns and mode == "fail":
        raise ValueError(f"Schema evolution failed: new columns detected: {new_columns}")
    
    type_changes = []
    for col_name in set(existing_fields.keys()) & set(new_fields.keys()):
        existing_type = existing_fields[col_name].dataType
        new_type = new_fields[col_name].dataType
        if existing_type != new_type:
            type_changes.append((col_name, existing_type, new_type))
    
    if type_changes and mode == "fail":
        raise ValueError(f"Schema evolution failed: type changes detected: {type_changes}")
    
    if mode == "merge":
        if new_columns:
            print(f"Adding new columns: {new_columns}")
            new_df.write.format("delta").mode("append").option("mergeSchema", "true").save(target_path)
        else:
            new_df.write.format("delta").mode("append").save(target_path)
    elif mode == "overwrite":
        new_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return True


def get_schema_diff(spark: SparkSession, table_path1: str, table_path2: str) -> Dict:
    df1 = spark.read.format("delta").load(table_path1)
    df2 = spark.read.format("delta").load(table_path2)
    
    schema1 = {f.name: str(f.dataType) for f in df1.schema.fields}
    schema2 = {f.name: str(f.dataType) for f in df2.schema.fields}
    
    return {
        "only_in_table1": set(schema1.keys()) - set(schema2.keys()),
        "only_in_table2": set(schema2.keys()) - set(schema1.keys()),
        "common_columns": set(schema1.keys()) & set(schema2.keys()),
        "type_differences": {
            col: (schema1[col], schema2[col])
            for col in set(schema1.keys()) & set(schema2.keys())
            if schema1[col] != schema2[col]
        }
    }


def validate_schema(spark: SparkSession, df: DataFrame, expected_schema: StructType) -> bool:
    actual_schema = df.schema
    
    if len(actual_schema.fields) != len(expected_schema.fields):
        return False
    
    for actual_field, expected_field in zip(actual_schema.fields, expected_schema.fields):
        if actual_field.name != expected_field.name:
            return False
        if actual_field.dataType != expected_field.dataType:
            return False
        if actual_field.nullable != expected_field.nullable:
            return False
    
    return True
