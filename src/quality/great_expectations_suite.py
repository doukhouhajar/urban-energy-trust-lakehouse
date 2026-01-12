from config.great_expectations import ExpectationSuite, ExpectationConfiguration
from typing import Dict, List


def create_halfhourly_consumption_suite() -> ExpectationSuite:
    suite = ExpectationSuite(
        expectation_suite_name="halfhourly_consumption_suite"
    )
    
    expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "household_id"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "timestamp"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "energy_kwh"}
        ),
        
        # Completeness
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "household_id"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "timestamp"}
        ),
        
        # Value ranges
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "energy_kwh",
                "min_value": 0.0,
                "max_value": 100.0
            }
        ),
        
        # Uniqueness
        ExpectationConfiguration(
            expectation_type="expect_compound_columns_to_be_unique",
            kwargs={
                "column_list": ["household_id", "timestamp"]
            }
        ),
        
        # Type checks
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={
                "column": "energy_kwh",
                "type_": "float64"
            }
        )
    ]
    
    for expectation in expectations:
        suite.add_expectation(expectation)
    
    return suite


def validate_with_great_expectations(
    df,
    suite: ExpectationSuite,
    evaluation_parameter_kwargs: Dict = None
):
    try:
        import great_expectations as ge
        
        # Convert Spark DataFrame to Pandas (for GE)
        pandas_df = df.toPandas()
        
        # Create GE DataFrame
        ge_df = ge.from_pandas(pandas_df)
        
        # Validate against suite
        validation_result = ge_df.validate(
            expectation_suite=suite,
            evaluation_parameters=evaluation_parameter_kwargs
        )
        
        return validation_result
        
    except ImportError:
        print("Warning: Great Expectations not installed. Skipping validation.")
        return None
    except Exception as e:
        print(f"Warning: Great Expectations validation failed: {e}")
        return None
