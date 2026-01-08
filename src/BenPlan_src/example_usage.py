#!/usr/bin/env python3
"""
Example usage of the Object-Oriented BenPlan classes

This script demonstrates how to use the new OO classes to process financial data
in a more modular and maintainable way.
"""

import sys
import asyncio
from benplan_oo import BenPlanProcessor, ConfigurationManager, DataLoader, DataCleaner, ReportGenerator, VisualizationManager


async def example_basic_usage():
    """Example of basic usage of the OO BenPlan classes."""
    print("=== Basic BenPlan Processing Example ===")
    
    # Create processor with command line arguments
    cmd_line = []  # No command line arguments for this example
    processor = BenPlanProcessor(cmd_line)
    
    # Process the data
    await processor.process()
    
    print("Basic processing completed!")


def example_individual_components():
    """Example of using individual components separately."""
    print("\n=== Individual Components Example ===")
    
    # Create configuration manager
    config_manager = ConfigurationManager("BenPlan", [])
    
    # Get current directory
    current_dir = config_manager.get_current_directory()
    print(f"Current directory: {current_dir}")
    
    # Get output paths
    paths = config_manager.get_output_paths()
    print(f"Output paths: {paths}")
    
    # Create data loader
    data_loader = DataLoader(config_manager)
    
    # Create data cleaner
    data_cleaner = DataCleaner(config_manager)
    
    # Create report generator
    report_generator = ReportGenerator(config_manager)
    
    # Create visualization manager
    viz_manager = VisualizationManager(config_manager)
    
    print("Individual components created successfully!")


def example_custom_processing():
    """Example of custom processing workflow."""
    print("\n=== Custom Processing Example ===")
    
    # Create configuration
    config_manager = ConfigurationManager("BenPlan", [])
    current_dir = config_manager.get_current_directory()
    
    # Create components
    data_loader = DataLoader(config_manager)
    data_cleaner = DataCleaner(config_manager)
    report_generator = ReportGenerator(config_manager)
    viz_manager = VisualizationManager(config_manager)
    
    # Custom processing workflow
    print("1. Loading transaction data...")
    # Note: In a real scenario, you'd need to create an output file
    # For this example, we'll just show the structure
    
    print("2. Cleaning and preparing data...")
    # cumul_df = data_cleaner.clean_and_prepare_data(cumul_df, current_dir, outf)
    
    print("3. Setting up report generation...")
    # conn1, write_process = report_generator.setup_async_writing()
    
    print("4. Generating visualizations...")
    # viz_manager.lineplot_df(df, title="Custom Analysis")
    
    print("Custom processing workflow demonstrated!")


async def example_async_processing():
    """Example of async processing capabilities."""
    print("\n=== Async Processing Example ===")
    
    # Create processor
    processor = BenPlanProcessor([])
    
    # Demonstrate async capabilities
    print("Starting async processing...")
    
    # In a real scenario, you might want to process multiple data sources concurrently
    tasks = []
    
    # Example: Process different data sources concurrently
    # task1 = asyncio.create_task(process_csv_files())
    # task2 = asyncio.create_task(process_excel_files())
    # task3 = asyncio.create_task(process_venmo_files())
    
    # tasks = [task1, task2, task3]
    # results = await asyncio.gather(*tasks)
    
    print("Async processing example completed!")


def example_configuration_management():
    """Example of configuration management."""
    print("\n=== Configuration Management Example ===")
    
    # Create configuration manager with different settings
    config_manager = ConfigurationManager("BenPlan", [])
    
    print(f"Program name: {config_manager.prog_name}")
    print(f"Birth month: {config_manager.birth_month}")
    print(f"Birth year: {config_manager.birth_year}")
    print(f"Window size: {config_manager.window}")
    print(f"BP simplified categories: {config_manager.bp_simpl_keep}")
    print(f"Discretionary categories: {config_manager.discretionary_cat}")
    
    # Access configuration settings
    print(f"Skip Venmo: {config_manager.default_config.get('SKIP_VENMO', False)}")
    print(f"Directory name pattern: {config_manager.default_config.get('dirNamePattern', 'Not set')}")


def main():
    """Main function to run all examples."""
    print("BenPlan Object-Oriented Refactoring Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_individual_components()
        example_configuration_management()
        example_custom_processing()
        
        # Run async examples
        asyncio.run(example_async_processing())
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey benefits of the OO refactoring:")
        print("1. Separation of concerns - each class has a specific responsibility")
        print("2. Better testability - individual components can be tested in isolation")
        print("3. Improved maintainability - changes to one component don't affect others")
        print("4. Reusability - components can be used independently")
        print("5. Better error handling - each class can handle its own errors")
        print("6. Easier to extend - new functionality can be added as new classes")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 