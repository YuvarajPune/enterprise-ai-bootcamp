import sqlite3


# -----------------------------
# HR Database
# -----------------------------

hr_db = sqlite3.connect("lessons/lesson01/hr.db")

hr_db.execute("""
CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT
)
""")

hr_db.execute("""
INSERT OR REPLACE INTO employees
(employee_id, name, department)
VALUES
(1, 'Raj', 'Engineering'),
(2, 'Priya', 'HR'),
(3, 'Amit', 'Finance')
""")

hr_db.commit()
hr_db.close()


# -----------------------------
# Finance Database
# -----------------------------

finance_db = sqlite3.connect("lessons/lesson01/finance.db")

finance_db.execute("""
CREATE TABLE IF NOT EXISTS salaries (
    employee_id INTEGER PRIMARY KEY,
    salary INTEGER
)
""")

finance_db.execute("""
INSERT OR REPLACE INTO salaries
(employee_id, salary)
VALUES
(1, 2500000),
(2, 1800000),
(3, 2200000)
""")

finance_db.commit()
finance_db.close()


print("HR and Finance databases created successfully.")