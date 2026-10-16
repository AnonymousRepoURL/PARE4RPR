# Tips for Creating New Rules 

- When adding new rules, please extend the base class `RewriteRule`

- Before deciding to add a new helper function, check if there are any readily available, reusable ones in `utils.py` or other python file.

- All characters are converted to uppercase, except for strings enclosed in quotation marks.

- After completing the rewrite rules, be sure to add optimizable queries for testing to facilitate the identification of code defects

- Add the rule name(s) to `__init__.py` to make it easier for external systems to call this rewrite rule

- Add the test module for the new rules to the `test_rewriting_rules.py` file to facilitate running all query tests.