import sqlite3


# Connect to the database
# If the database does not exist, SQLite will create it
connection = sqlite3.connect("database/company.db")

cursor = connection.cursor()


# Create employee table
cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    designation TEXT NOT NULL,
    salary REAL
)
""")


# Sample employee data
employees = [
    ("EMP001", "Ravi Kumar", "Engineering", "Software Engineer", 75000),
    ("EMP002", "Anita Sharma", "HR", "HR Manager", 85000),
    ("EMP003", "Rahul Reddy", "Engineering", "Data Scientist", 95000),
    ("EMP004", "Priya Singh", "Finance", "Financial Analyst", 80000),
    ("EMP005", "Arjun Patel", "Engineering", "AI Engineer", 110000)
]


# Insert employees
cursor.executemany("""
INSERT OR IGNORE INTO employees
(employee_id, name, department, designation, salary)
VALUES (?, ?, ?, ?, ?)
""", employees)


# Save changes
connection.commit()


# Close connection
connection.close()


print("Database created successfully!")