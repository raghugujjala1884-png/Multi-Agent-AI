import sqlite3


connection = sqlite3.connect("database/company.db")

cursor = connection.cursor()

cursor.execute("""
    SELECT employee_id, name, department, designation, salary
    FROM employees
""")

employees = cursor.fetchall()

connection.close()


print("\nEMPLOYEES IN DATABASE")
print("=" * 60)

for employee in employees:
    print(employee)