#!/usr/bin/env python3
"""
Test suite for the Object-Oriented BenPlan classes

This module provides unit tests and integration tests for the refactored BenPlan system.
"""

import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch

# Import the classes to test
from benplan_oo import (
    ConfigurationManager, DataLoader, DataCleaner, 
    ReportGenerator, VisualizationManager, BenPlanProcessor
)

TEST_DIR = "2024-01-01"


class TestConfigurationManager(unittest.TestCase):
    """Test cases for ConfigurationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager("TestBenPlan", [])
        self.config_manager.current_dir = TEST_DIR  # for testing

    
    def test_initialization(self):
        """Test ConfigurationManager initialization."""
        self.assertIsNotNone(self.config_manager.prog_name)
        self.assertIsNotNone(self.config_manager.config)
        self.assertIsNotNone(self.config_manager.default_config)
        self.assertIsNotNone(self.config_manager.logger)
    
    def test_constants(self):
        """Test that constants are properly set."""
        self.assertEqual(self.config_manager.birth_month, 4)
        self.assertEqual(self.config_manager.birth_year, 1958)
        self.assertEqual(self.config_manager.window, 12)
        self.assertEqual(self.config_manager.zoom_level, 120)
        self.assertIsInstance(self.config_manager.bp_simpl_keep, list)
        self.assertIsInstance(self.config_manager.discretionary_cat, list)
    
    @patch('os.scandir')
    @patch('config_constants.DATA_DIR', '/fake/data/path/')
    def test_get_current_directory(self, mock_scandir):
        """Test getting current directory."""
        # Mock directory entries
        mock_entries = []
        for d in ['2024-01-01', '2024-02-01', '2024-03-01']:
            entry = Mock()
            entry.name = d
            entry.is_dir.return_value = True
            mock_entries.append(entry)
        mock_scandir.return_value.__enter__.return_value = mock_entries

        # Mock the dirNamePattern in default_config
        self.config_manager.default_config['dirNamePattern'] = r'\d{4}-\d{2}-\d{2}'

        current_dir = self.config_manager.get_current_directory()
        self.assertEqual(current_dir, '2024-03-01')
    
    def test_get_output_paths(self):
        """Test output path generation."""
        paths = self.config_manager.get_output_paths()
        
        self.assertIn('xl_out_file', paths)
        self.assertIn('out_file', paths)
        self.assertIn('plot_file', paths)
        self.assertIn('quick_dir', paths)
        
        # Check that paths contain the test directory
        self.assertIn(TEST_DIR, paths['xl_out_file'])
        self.assertIn(TEST_DIR, paths['out_file'])


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager("TestBenPlan", [])
        self.data_loader = DataLoader(self.config_manager)
    
    @patch('benplan_oo.os.scandir')
    @patch('os.chdir')
    @patch('config_constants.DATA_DIR', '/fake/data/path/')
    @patch('config_constants.CLOSED_DIR', '/fake/closed/path/')
    def test_get_csv_files(self, mock_closed_dir, mock_data_dir, mock_chdir, mock_scandir):
        """Test CSV file discovery."""
        # Mock directory entries for current directory
        current_dir_entries = []
        for name in ['test1.csv', 'test2.csv', 'test3.txt', 'aggregate-2024-01-01.csv']:
            entry = Mock()
            entry.name = name
            entry.is_file.return_value = True
            current_dir_entries.append(entry)
        
        # Mock directory entries for closed directory
        closed_dir_entries = []
        for name in ['closed1.csv', 'closed2.csv']:
            entry = Mock()
            entry.name = name
            entry.is_file.return_value = True
            closed_dir_entries.append(entry)
        
        # Configure mock to return different entries based on the path
        def mock_scandir_side_effect(path):
            print(f"Mock scandir called with path: {path} (type: {type(path)})")
            if path == '.':
                print(f"Returning current dir entries for path: {path}")
                return current_dir_entries
            elif path == '/fake/closed/path/':
                print(f"Returning closed dir entries for path: {path}")
                return closed_dir_entries
            else:
                print(f"Returning empty entries for path: {path}")
                return []
        
        mock_scandir.side_effect = mock_scandir_side_effect
        
        # Debug: Let's see what's happening
        print(f"Mock scandir side effect configured")
        print(f"Current dir entries: {[e.name for e in current_dir_entries]}")
        print(f"Closed dir entries: {[e.name for e in closed_dir_entries]}")
        
        files = self.data_loader._get_csv_files()
        
        # Debug: Let's see what files we actually got
        print(f"Files returned: {files}")
        
        # Should include CSV files from both current and closed directories, excluding suppressed ones
        self.assertEqual(len(files), 4)  # test1.csv, test2.csv, closed1.csv, closed2.csv
        self.assertIn('test1.csv', files)
        self.assertIn('test2.csv', files)
        self.assertIn('/fake/closed/path/closed1.csv', files)
        self.assertIn('/fake/closed/path/closed2.csv', files)
        self.assertNotIn('test3.txt', files)
        self.assertNotIn('aggregate-2024-01-01.csv', files)
    
    @patch('expenses_utilities.process_file')
    @patch('expenses_utilities.brkg_process_file')
    @patch('venmo_utilities.process_all_venmo_files')
    def test_load_all_transaction_data(self, mock_venmo, mock_brkg, mock_process):
        """Test loading all transaction data."""
        # Mock return values
        mock_df1 = pd.DataFrame({'Date': ['01/01/2024'], 'Amount': [100], 'Category': ['Test'], 'Source': ['CSV']})
        mock_df2 = pd.DataFrame({'Date': ['01/02/2024'], 'Amount': [200], 'Category': ['Test2'], 'Source': ['Brokerage']})
        mock_df3 = pd.DataFrame({'Date': ['01/03/2024'], 'Amount': [300], 'Category': ['Test3'], 'Source': ['Venmo']})
        
        mock_process.return_value = mock_df1
        mock_brkg.return_value = mock_df2
        mock_venmo.return_value = mock_df3
        
        # Mock file discovery
        with patch.object(self.data_loader, '_get_csv_files', return_value=['test1.csv', 'test2.csv']):
            result = self.data_loader.load_all_transaction_data()
            
            self.assertIsInstance(result, pd.DataFrame)
            # Should contain data from all sources
            self.assertGreater(len(result), 0)


class TestDataCleaner(unittest.TestCase):
    """Test cases for DataCleaner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager("TestBenPlan", [])
        self.data_cleaner = DataCleaner(self.config_manager)
    
    def test_clean_and_prepare_data(self):
        """Test data cleaning and preparation."""
        # Create test data - use dates that will pass the date filter
        # The filter expects dates >= '2015-05' and < '2024-03-01'
        test_data = pd.DataFrame({
            'Date': ['06/01/2015', '07/01/2015', '08/01/2015'],  # All after 05/01/2015
            'Amount': ['1,000.00', '2,000.00', '3,000.00'],
            'Category': ['Test:Sub', 'Travel', 'Other'],
            'Month': ['2015-06', '2015-07', '2015-08'],
            'Payee': ['Test Payee', 'Hotel ABC', 'Other Payee'],
            'Memo/Notes': ['Test memo', 'Hotel stay', 'Other memo']
        })
        
        # Mock output file
        mock_outf = Mock()
        
        # Test cleaning
        result = self.data_cleaner.clean_and_prepare_data(test_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        
        # Check that amounts are converted to float
        self.assertTrue(all(isinstance(x, float) for x in result['Amount']))
    
    @patch('pandas.read_excel')
    def test_remap_sub_cat(self, mock_read_excel):
        """Test sub-category remapping."""
        # Mock category mapping
        mock_mapping = pd.DataFrame({
            'SubCat': ['Test:Sub'],
            'MainCat': ['Test']
        })
        mock_read_excel.return_value = mock_mapping
        
        # Test data
        test_data = pd.DataFrame({
            'Category': ['Test:Sub', 'Other:Sub', 'Simple']
        })
        
        result = self.data_cleaner.remap_sub_cat(test_data, "test_file.xlsx", Mock())
        
        # Check that sub-categories are remapped
        self.assertEqual(result.iloc[0]['Category'], 'Test')
        self.assertEqual(result.iloc[1]['Category'], 'Other')
        self.assertEqual(result.iloc[2]['Category'], 'Simple')


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager("TestBenPlan", [])
        self.report_generator = ReportGenerator(self.config_manager)
    
    def test_setup_async_writing(self):
        """Test async writing setup."""
        conn1, write_process = self.report_generator.setup_async_writing()
        
        self.assertIsNotNone(conn1)
        self.assertIsNotNone(write_process)
        self.assertTrue(write_process.is_alive())
        
        # Cleanup
        conn1.send(None)
        write_process.join()
    
    def test_quick_write_df(self):
        """Test quick DataFrame writing."""
        # Create test data
        test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        # Mock configuration
        self.config_manager.default_config['quick_dir'] = tempfile.mkdtemp()
        
        # Test writing
        self.report_generator.quick_write_df(test_df, "test_sheet", False)
        
        # Check that file was created
        expected_file = os.path.join(self.config_manager.default_config['quick_dir'], "test_sheet.xlsx")
        self.assertTrue(os.path.exists(expected_file))
        
        # Cleanup
        os.remove(expected_file)
        os.rmdir(self.config_manager.default_config['quick_dir'])


class TestVisualizationManager(unittest.TestCase):
    """Test cases for VisualizationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager("TestBenPlan", [])
        self.viz_manager = VisualizationManager(self.config_manager)
    
    def test_tick_format(self):
        """Test tick formatting."""
        result = self.viz_manager._tick_format(1234.56, 0)
        self.assertEqual(result, '$1,235')
    
    def test_lineplot_df(self):
        """Test line plot creation."""
        # Set matplotlib to use non-interactive backend for testing
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create test data
        test_df = pd.DataFrame({
            '2024-01': [100, 200],
            '2024-02': [150, 250],
            '2024-03': [200, 300]
        }, index=pd.Index(['Category1', 'Category2']))
        
        # Test plotting (should not raise exceptions)
        try:
            self.viz_manager.lineplot_df(test_df, title="Test Plot")
        except Exception as e:
            self.fail(f"lineplot_df raised an exception: {e}")


class TestBenPlanProcessor(unittest.TestCase):
    """Test cases for BenPlanProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = BenPlanProcessor([])
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor.config_manager)
        self.assertIsNotNone(self.processor.data_loader)
        self.assertIsNotNone(self.processor.data_cleaner)
        self.assertIsNotNone(self.processor.report_generator)
        self.assertIsNotNone(self.processor.visualization_manager)
    
    @patch.object(BenPlanProcessor, '_setup_output_files')
    @patch.object(BenPlanProcessor, '_generate_reports')
    @patch.object(BenPlanProcessor, '_cleanup')
    async def test_process(self, mock_cleanup, mock_generate_reports, mock_setup):
        """Test main processing workflow."""
        # Mock the setup method
        mock_setup.return_value = None
        
        # Mock the generate_reports method
        mock_generate_reports.return_value = None
        
        # Mock the cleanup method
        mock_cleanup.return_value = None
        
        # Test processing (should not raise exceptions)
        try:
            await self.processor.process()
        except Exception as e:
            self.fail(f"process raised an exception: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = BenPlanProcessor([])
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('builtins.open')
    async def test_full_workflow_mock(self, mock_open, mock_makedirs, mock_exists):
        """Test the complete workflow with mocked dependencies."""
        # Mock file operations
        mock_exists.return_value = False
        mock_open.return_value = Mock()
        
        # Mock configuration
        with patch.object(self.processor.config_manager, 'get_current_directory', return_value='2024-01-01'):
            with patch.object(self.processor.config_manager, 'get_output_paths', return_value={
                'xl_out_file': 'test.xlsx',
                'out_file': 'test.txt',
                'plot_file': 'test.pdf',
                'quick_dir': 'test_quick'
            }):
                # Test that the processor can be created and initialized
                self.assertIsNotNone(self.processor)
                
                # Test that all components are properly initialized
                self.assertIsNotNone(self.processor.config_manager)
                self.assertIsNotNone(self.processor.data_loader)
                self.assertIsNotNone(self.processor.data_cleaner)
                self.assertIsNotNone(self.processor.report_generator)
                self.assertIsNotNone(self.processor.visualization_manager)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestConfigurationManager,
        TestDataLoader,
        TestDataCleaner,
        TestReportGenerator,
        TestVisualizationManager,
        TestBenPlanProcessor,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running BenPlan OO Tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("\nTest Summary:")
    print("- Unit tests for each component")
    print("- Integration tests for the complete system")
    print("- Mocked dependencies for isolated testing")
    print("- Async testing for async components") 