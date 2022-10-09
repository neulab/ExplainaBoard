# Text-to-SQL with Test Suites

## Source 
This folder contains third-party codes, adapted from https://github.com/taoyds/test-suite-sql-eval, for the text-to-SQL task evaluation.

The original code is obtained and added on 10/4/2022.

`evaluation.py`, `exec_eval.py`, `parse.py`, `process_sql.py` are from the original repository, and `evaluation_test.py` is the unit test file we added. 

## Modifications
The input of `sql_evaluate` function in `evaluation.py` is modified with combining multiple configuration arguments into a dictionary.  
